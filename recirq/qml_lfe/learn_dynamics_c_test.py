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

import os
import pytest
import numpy as np
from . import learn_dynamics_c


def test_shadows_dont_seperate(tmpdir):
    """Ensure that classical shadows cannot distinguish tsym vs scrambling circuits."""
    learn_dynamics_c.run_and_save(
        n=6,
        depth=5,
        n_data=20,
        batch_size=20,
        n_shots=2000,
        save_dir=tmpdir,
        use_engine=False,
    )

    tsym_datapoints = []
    scramble_datapoints = []
    for i in range(20):
        fname = f"1D-scramble-C-size-6-depth-5-type-1-batch-0-number-{i}.npy"
        t = np.load(os.path.join(tmpdir, fname))
        tsym_datapoints.append([np.mean(t, axis=0), np.std(t, axis=0)])
        fname = f"1D-scramble-C-size-6-depth-5-type-0-batch-0-number-{i}.npy"
        t = np.load(os.path.join(tmpdir, fname))
        scramble_datapoints.append([np.mean(t, axis=0), np.std(t, axis=0)])

    scramble_bitwise_stats = np.mean(scramble_datapoints, axis=0)
    tsym_bitwise_stats = np.mean(tsym_datapoints, axis=0)
    expected_diff = np.zeros_like(tsym_bitwise_stats)
    # Should see no meaningful difference between measurement stats
    # when using shadows.
    np.testing.assert_allclose(
        scramble_bitwise_stats - tsym_bitwise_stats, expected_diff, atol=0.15
    )
