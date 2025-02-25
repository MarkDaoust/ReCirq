# Copyright 2021 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Draw two copy data from 1D random scrambling or tsym style circuits.

learn_dynamics_q.py --n=6 --depth=5 --n_data=20 --batch_size=5 --n_shots=1000

Will create 20 scrambling and 20 tsym circuits (40 total circuits) all on
12 qubits with depth 5. Once the circuits are generated, batch_size circuits
will be sent for simulation/execution, drawing n_shots samples from each
one using `run_batch` in Cirq. By default the bitstring data will be saved
in the data folder. One can also set `use_engine` to True in order to
run this against a processor on quantum engine.
"""

from typing import List

import os
import cirq
import numpy as np
from . import circuit_blocks
from . import run_config
from absl import app
from absl import logging


def _build_circuit(
    qubit_pairs: List[List[cirq.Qid]], use_tsym: bool, depth: int
) -> cirq.Circuit:
    """Generate twocopy 1D tsym or scrambling circuit with given depth on qubit_pairs."""
    inter_gen = circuit_blocks.scrambling_block
    if use_tsym:
        # Use tsym gates.
        inter_gen = circuit_blocks.tsym_block

    # Random source for circuits.
    random_source = np.random.uniform(0, 4, size=(depth * len(qubit_pairs), 2))

    # Create initial bell pair circuit.
    ret_circuit = cirq.Circuit(
        [circuit_blocks.bell_pair_block(pair) for pair in qubit_pairs]
    )

    # Add tsym or scrambling block on System A qubits.
    ret_circuit += circuit_blocks.block_1d_circuit(
        [qbpair[0] for qbpair in qubit_pairs], depth, inter_gen, random_source
    )

    # Swap system A and system B qubits.
    ret_circuit += [circuit_blocks.swap_block(pair) for pair in qubit_pairs]

    # Again run tsym or scrambling block on newly swapped qubits.
    ret_circuit += circuit_blocks.block_1d_circuit(
        [qbpair[0] for qbpair in qubit_pairs], depth, inter_gen, random_source
    )

    # Un bell pair.
    ret_circuit += [circuit_blocks.un_bell_pair_block(pair) for pair in qubit_pairs]

    # Merge single qubit gates together and add measurements.
    cirq.merge_single_qubit_gates_into_phxz(ret_circuit)
    cirq.DropEmptyMoments().optimize_circuit(circuit=ret_circuit)
    ret_circuit = run_config.flatten_circuit(ret_circuit)

    for i, qubit in enumerate([item for sublist in qubit_pairs for item in sublist]):
        ret_circuit += cirq.measure(qubit, key="q{}".format(i))

    cirq.SynchronizeTerminalMeasurements().optimize_circuit(circuit=ret_circuit)
    logging.debug(
        f"Generated a new circuit w/ tsym={use_tsym} and depth {len(ret_circuit)}"
    )
    return ret_circuit


def run_and_save(
    n: int,
    depth: int,
    n_data: int,
    batch_size: int,
    n_shots: int,
    save_dir: str,
    use_engine: bool,
) -> None:
    """Run and save bitstring data for tsym vs scrambling experiment w/ twocopy.

    Note: uses qubit layouts native to a Sycamore device (Weber).

    Args:
        n: Number of system qubits to use (total qubits == 2 * n).
        depth: Number of tsym or scrambling blocks.
        n_data: Number of tsym and scrambling circuits to generate.
        batch_size: The number of circuits to send off for execution in
            a single payload (Does not affect experimental resluts).
        save_dir: str or Path to directory where data is saved.
        use_engine: Whether or not to make use of quantum engine and the
            weber processor. Note this requires the GOOGLE_CLOUD_PROJECT
            environment variable to be set, along with the required cloud
            permissions.
    """
    logging.info("Beginning quantum-enhanced circuit generation.")
    # Choose system pairs so that they are consecutive neighbors.
    system_pairs = run_config.qubit_pairs()
    system_pairs = system_pairs[:n]

    to_run_scramb = [_build_circuit(system_pairs, False, depth) for _ in range(n_data)]
    to_run_tsym = [_build_circuit(system_pairs, True, depth) for _ in range(n_data)]
    logging.info(
        "Circuit generation complete."
        f"Generated {len(to_run_tsym) + len(to_run_scramb)}"
        "total circuits"
    )
    for k in range(0, n_data, batch_size):
        logging.info(f"Running batch: [{k}-{k + batch_size}) / {n_data}")
        for is_tsym in [0, 1]:

            # Alternate between scrambling and tsym to
            # mitigate drift related noise artifacts.
            batch = to_run_scramb[k : k + batch_size]
            if is_tsym:
                batch = to_run_tsym[k : k + batch_size]

            # Upload and run the circuit batch.
            results = run_config.execute_batch(batch, n_shots, use_engine)

            for j, single_circuit_samples in enumerate(results):
                name0 = (
                    f"1D-scramble-Q-size-{n}"
                    f"-depth-{depth}"
                    f"-type-{is_tsym}"
                    f"-batch-{k}"
                    f"-number-{j}"
                )
                qubit_order = ["q{}".format(i) for i in range(n * 2)]
                out0 = single_circuit_samples.data[qubit_order].to_numpy()
                np.save(os.path.join(save_dir, name0), out0)
                logging.debug("Saved: " + name0)


def main(_):
    from . import dynamics_flags

    run_and_save(
        dynamics_flags.n,
        dynamics_flags.depth,
        dynamics_flags.n_data,
        dynamics_flags.batch_size,
        dynamics_flags.n_shots,
        dynamics_flags.save_dir,
        dynamics_flags.use_engine,
    )


if __name__ == "__main__":
    app.run(main)
