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
        Vectorized VLIW kernel - processes VLEN (8) elements per iteration.
        Optimized with better VLIW packing.
        """
        # Scalar temporaries
        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Scratch space addresses for memory layout
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

        # Scalar constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Preload all hash constants before the loop
        hash_consts = []
        for (op1, val1, op2, op3, val3) in HASH_STAGES:
            hash_consts.append((self.scratch_const(val1), self.scratch_const(val3)))

        # Vector scratch registers (VLEN = 8 elements each)
        v_idx = self.alloc_scratch("v_idx", VLEN)
        v_val = self.alloc_scratch("v_val", VLEN)
        v_node_val = self.alloc_scratch("v_node_val", VLEN)
        v_tmp1 = self.alloc_scratch("v_tmp1", VLEN)
        v_tmp2 = self.alloc_scratch("v_tmp2", VLEN)
        v_tmp3 = self.alloc_scratch("v_tmp3", VLEN)
        v_addr = self.alloc_scratch("v_addr", VLEN)

        # Scalar address registers
        addr_idx = self.alloc_scratch("addr_idx")
        addr_val = self.alloc_scratch("addr_val")

        # Vector constants
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)

        # Broadcast scalar constants to vectors - pack 2 per cycle
        self.add_vliw([
            ("valu", ("vbroadcast", v_zero, zero_const)),
            ("valu", ("vbroadcast", v_one, one_const)),
        ])
        self.add_vliw([
            ("valu", ("vbroadcast", v_two, two_const)),
            ("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"])),
        ])

        # Precompute hash constant vectors - pack 2 broadcasts per cycle
        v_hash_consts = []
        for (c1, c3) in hash_consts:
            vc1 = self.alloc_scratch(None, VLEN)
            vc3 = self.alloc_scratch(None, VLEN)
            self.add_vliw([
                ("valu", ("vbroadcast", vc1, c1)),
                ("valu", ("vbroadcast", vc3, c3)),
            ])
            v_hash_consts.append((vc1, vc3))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting vectorized loop"))

        n_vec_iters = batch_size // VLEN

        for round in range(rounds):
            for vi in range(n_vec_iters):
                base_i = vi * VLEN
                base_const = self.scratch_const(base_i)

                # PACK: Calculate base addresses + vbroadcast forest_values_p
                self.add_vliw([
                    ("alu", ("+", addr_idx, self.scratch["inp_indices_p"], base_const)),
                    ("alu", ("+", addr_val, self.scratch["inp_values_p"], base_const)),
                    ("valu", ("vbroadcast", v_addr, self.scratch["forest_values_p"])),
                ])

                # Vector load idx and val
                self.add_vliw([
                    ("load", ("vload", v_idx, addr_idx)),
                    ("load", ("vload", v_val, addr_val)),
                ])

                self.add_vliw([
                    ("debug", ("vcompare", v_idx, tuple((round, base_i + j, "idx") for j in range(VLEN)))),
                    ("debug", ("vcompare", v_val, tuple((round, base_i + j, "val") for j in range(VLEN)))),
                ])

                # Compute gather addresses
                self.add_vliw([("valu", ("+", v_addr, v_addr, v_idx))])

                # Gather with multiply packed in first cycle
                self.add_vliw([
                    ("load", ("load_offset", v_node_val, v_addr, 0)),
                    ("load", ("load_offset", v_node_val, v_addr, 1)),
                    ("valu", ("*", v_idx, v_idx, v_two)),
                ])
                for lo in range(2, VLEN, 2):
                    self.add_vliw([
                        ("load", ("load_offset", v_node_val, v_addr, lo)),
                        ("load", ("load_offset", v_node_val, v_addr, lo + 1)),
                    ])

                self.add_vliw([
                    ("debug", ("vcompare", v_node_val, tuple((round, base_i + j, "node_val") for j in range(VLEN)))),
                ])

                # XOR before hash
                self.add_vliw([("valu", ("^", v_val, v_val, v_node_val))])

                # Vector hash function
                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    vc1, vc3 = v_hash_consts[hi]
                    self.add_vliw([
                        ("valu", (op1, v_tmp1, v_val, vc1)),
                        ("valu", (op3, v_tmp2, v_val, vc3)),
                    ])
                    self.add_vliw([("valu", (op2, v_val, v_tmp1, v_tmp2))])
                    self.add_vliw([
                        ("debug", ("vcompare", v_val, tuple((round, base_i + j, "hash_stage", hi) for j in range(VLEN)))),
                    ])

                self.add_vliw([
                    ("debug", ("vcompare", v_val, tuple((round, base_i + j, "hashed_val") for j in range(VLEN)))),
                ])

                # idx = 2*idx + (1 if val % 2 == 0 else 2)
                # Note: v_idx * 2 was already computed during gather
                self.add_vliw([("valu", ("%", v_tmp1, v_val, v_two))])
                self.add_vliw([("valu", ("==", v_tmp1, v_tmp1, v_zero))])
                self.add_vliw([("flow", ("vselect", v_tmp3, v_tmp1, v_one, v_two))])
                self.add_vliw([("valu", ("+", v_idx, v_idx, v_tmp3))])
                self.add_vliw([
                    ("debug", ("vcompare", v_idx, tuple((round, base_i + j, "next_idx") for j in range(VLEN)))),
                ])

                # idx = 0 if idx >= n_nodes else idx
                self.add_vliw([("valu", ("<", v_tmp1, v_idx, v_n_nodes))])
                self.add_vliw([("flow", ("vselect", v_idx, v_tmp1, v_idx, v_zero))])
                self.add_vliw([
                    ("debug", ("vcompare", v_idx, tuple((round, base_i + j, "wrapped_idx") for j in range(VLEN)))),
                ])

                # Vector store
                self.add_vliw([
                    ("store", ("vstore", addr_idx, v_idx)),
                    ("store", ("vstore", addr_val, v_val)),
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
