"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Pack slots into VLIW instruction bundles respecting slot limits
        if not vliw:
            # Original behavior: one slot per instruction
            instrs = []
            for engine, slot in slots:
                instrs.append({engine: [slot]})
            return instrs

        # VLIW packing: group slots into bundles respecting limits
        instrs = []
        current_instr = {}
        current_counts = defaultdict(int)

        for engine, slot in slots:
            if engine == "debug":
                # Debug instructions don't count toward cycles, add to current
                if engine not in current_instr:
                    current_instr[engine] = []
                current_instr[engine].append(slot)
                continue

            # Check if we can add this slot to the current instruction
            if current_counts[engine] < SLOT_LIMITS[engine]:
                if engine not in current_instr:
                    current_instr[engine] = []
                current_instr[engine].append(slot)
                current_counts[engine] += 1
            else:
                # Start a new instruction
                if current_instr:
                    instrs.append(current_instr)
                current_instr = {engine: [slot]}
                current_counts = defaultdict(int)
                current_counts[engine] = 1

        if current_instr:
            instrs.append(current_instr)

        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_vliw(self, slots):
        """Add multiple slots as a single VLIW instruction bundle."""
        instr = defaultdict(list)
        for engine, slot in slots:
            instr[engine].append(slot)
        # Verify slot limits
        for engine, slot_list in instr.items():
            if engine != "debug":
                assert len(slot_list) <= SLOT_LIMITS[engine], f"Too many {engine} slots: {len(slot_list)}"
        self.instrs.append(dict(instr))

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i, vliw=False):
        slots = []

        if not vliw:
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
                slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
                slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
                slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))
        else:
            # VLIW-optimized: group the two independent ops together
            for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                # These two operations are independent (both read val_hash_addr, write to different regs)
                slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
                slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
                # Mark end of parallel group - this depends on tmp1 and tmp2
                slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
                slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Like reference_kernel2 but building actual instructions.
        VLIW-optimized: pack independent operations into parallel slots.
        """
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")
        # Scratch space addresses
        init_vars = [
            "rounds",
            "n_nodes",
            "batch_size",
            "forest_height",
            "forest_values_p",
            "inp_indices_p",
            "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pause instructions are matched up with yield statements in the reference
        # kernel to let you debug at intermediate steps. The testing harness in this
        # file requires these match up to the reference kernel's yields, but the
        # submission harness ignores them.
        self.add("flow", ("pause",))
        # Any debug engine instruction is ignored by the submission simulator
        self.add("debug", ("comment", "Starting loop"))

        # Scalar scratch registers - allocate TWO address registers for parallel loads
        tmp_idx = self.alloc_scratch("tmp_idx")
        tmp_val = self.alloc_scratch("tmp_val")
        tmp_node_val = self.alloc_scratch("tmp_node_val")
        tmp_addr1 = self.alloc_scratch("tmp_addr")  # For idx address
        tmp_addr2 = self.alloc_scratch("tmp_addr2")  # For val address

        for round in range(rounds):
            for i in range(batch_size):
                i_const = self.scratch_const(i)

                # VLIW: Calculate both addresses in parallel (2 ALU slots)
                self.add_vliw([
                    ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], i_const)),
                    ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)),
                ])

                # VLIW: Load idx and val in parallel (2 LOAD slots)
                self.add_vliw([
                    ("load", ("load", tmp_idx, tmp_addr1)),
                    ("load", ("load", tmp_val, tmp_addr2)),
                ])
                # Debug compares must be in separate instruction (reads happen before writes commit)
                self.add_vliw([
                    ("debug", ("compare", tmp_idx, (round, i, "idx"))),
                    ("debug", ("compare", tmp_val, (round, i, "val"))),
                ])

                # node_val = mem[forest_values_p + idx] (sequential - depends on tmp_idx)
                self.add_vliw([("alu", ("+", tmp_addr1, self.scratch["forest_values_p"], tmp_idx))])
                self.add_vliw([("load", ("load", tmp_node_val, tmp_addr1))])
                self.add_vliw([("debug", ("compare", tmp_node_val, (round, i, "node_val")))])

                # val = myhash(val ^ node_val)
                self.add_vliw([("alu", ("^", tmp_val, tmp_val, tmp_node_val))])

                # Hash function with VLIW packing
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # These two ops are independent - pack them together
                    self.add_vliw([
                        ("alu", (op1, tmp1, tmp_val, self.scratch_const(val1))),
                        ("alu", (op3, tmp2, tmp_val, self.scratch_const(val3))),
                    ])
                    # This depends on both tmp1 and tmp2
                    self.add_vliw([("alu", (op2, tmp_val, tmp1, tmp2))])
                    self.add_vliw([("debug", ("compare", tmp_val, (round, i, "hash_stage", hi)))])

                self.add_vliw([("debug", ("compare", tmp_val, (round, i, "hashed_val")))])

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # Can pack modulo and multiply in parallel (different dest regs)
                self.add_vliw([
                    ("alu", ("%", tmp1, tmp_val, two_const)),
                    ("alu", ("*", tmp_idx, tmp_idx, two_const)),
                ])
                self.add_vliw([("alu", ("==", tmp1, tmp1, zero_const))])
                self.add_vliw([("flow", ("select", tmp3, tmp1, one_const, two_const))])
                self.add_vliw([("alu", ("+", tmp_idx, tmp_idx, tmp3))])
                self.add_vliw([("debug", ("compare", tmp_idx, (round, i, "next_idx")))])

                # idx = 0 if idx >= n_nodes else idx
                self.add_vliw([("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"]))])
                self.add_vliw([("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const))])
                self.add_vliw([("debug", ("compare", tmp_idx, (round, i, "wrapped_idx")))])

                # VLIW: Calculate both store addresses in parallel
                self.add_vliw([
                    ("alu", ("+", tmp_addr1, self.scratch["inp_indices_p"], i_const)),
                    ("alu", ("+", tmp_addr2, self.scratch["inp_values_p"], i_const)),
                ])

                # VLIW: Store both values in parallel (2 STORE slots)
                self.add_vliw([
                    ("store", ("store", tmp_addr1, tmp_idx)),
                    ("store", ("store", tmp_addr2, tmp_val)),
                ])

        # Required to match with the yield in reference_kernel2
        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
