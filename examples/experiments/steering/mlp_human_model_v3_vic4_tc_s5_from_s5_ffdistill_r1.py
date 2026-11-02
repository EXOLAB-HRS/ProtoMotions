# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""vic4_TC_s5_from-s5_ffdistill_r1 (cleaned law, server, 2026-11-03; user: "좀 더 깔끔하게", approved night plan):
vic4c_TC_s5_from-s5_ffdistill_r1 (s5 -> human impedance centre at once, alpha 1 / beta 0 / no schedule / no beta prior,
no VIC effort, feed-forward distillation from the frozen fixed-PD teacher se-v1-s5, swing knee out of r_imp) with one
change: the stiffness outputs are unfrozen (v4_freeze_a False), so the law is
    tau = tau_imp(k(a), b(k)) + LPF30ms(tau_res),   tau_res pulled toward (teacher PD torque - own PD torque)
and the fast torque the 30 Hz feed-forward cannot follow goes to stiffness modulation inside [k_lo / 2, 2 k_hi]
(tau_ff diagnostic 2026-11-02: 75-97 % of the fixed-PD -> human-impedance torque gap is below 5 Hz; the fast part is
>= 90 % coverable by in-range stiffness at hip / ankle, 53-63 % at the knee). Used on plant human_model_v3.3 (launcher
--plant v3.3), from se-v1-s5 or from the v3.3-retuned walker.
"""
import dataclasses

from examples.experiments.steering import mlp_human_model_v3_vic4c_tc_s5_from_s5_ffdistill_r1 as _ffd
from examples.experiments.steering.mlp_human_model_v3_vic4c_tc_s5_from_s5_ffdistill_r1 import *  # noqa: F401,F403


def env_config(robot_cfg, args):
    cfg = _ffd.env_config(robot_cfg, args)
    cfg.control_components["steering"] = dataclasses.replace(cfg.control_components["steering"], v4_freeze_a=False)
    return cfg


def agent_config(robot_cfg, env_cfg, args):
    return _ffd.agent_config(robot_cfg, env_cfg, args)
