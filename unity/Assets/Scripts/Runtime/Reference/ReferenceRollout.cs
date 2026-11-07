using System;
using System.IO;
using UnityEngine;

namespace BMB
{
    // Field names match the JSON keys written by export/npz_to_json.py (JsonUtility needs exact names),
    // hence snake_case. Field-by-field description: experiments/w3_reference/README.md.
#pragma warning disable IDE1006, CS0649

    /// <summary>Array stored as {"shape": [...], "data": [...]} flattened row-major (C order).</summary>
    [Serializable]
    public sealed class NdArray
    {
        public int[] shape;
        public float[] data;

        public bool IsEmpty => data == null || data.Length == 0;
        public int Rows => shape[0];
        public int RowSize => data.Length / shape[0];

        public float this[int row, int col] => data[row * RowSize + col];

        /// <summary>Copy of row t (all trailing dimensions flattened).</summary>
        public float[] Row(int t)
        {
            var row = new float[RowSize];
            Array.Copy(data, t * RowSize, row, 0, RowSize);
            return row;
        }
    }

    /// <summary>
    /// Isaac Lab reference rollout (500 policy steps, fixed command). World frame z-up, quaternions (w, x, y, z).
    /// State fields are captured before action t is applied (what obs[t] was computed from);
    /// actions / joint_pos_target / applied_torque after the step.
    /// </summary>
    [Serializable]
    public sealed class ReferenceRollout
    {
        public NdArray obs, actions, command, joint_pos, joint_vel;
        public NdArray root_link_pos_w, root_link_quat_w, root_com_lin_vel_w, root_com_ang_vel_w;
        public NdArray body_link_pos_w, body_link_quat_w;
        public NdArray joint_pos_target, applied_torque, dones;
        public NdArray scan_sensor_z, scan_hits_z, scan_hits_xy;  // rough / v2 only
        public NdArray command_fixed, default_joint_pos, stiffness, damping, effort_limit, velocity_limit;
        public NdArray body_mass, body_inertia, body_com_pose;
        public NdArray joint_pos_limits, joint_vel_limits, joint_effort_limits, joint_armature, joint_friction_coeff;
        public string task, checkpoint, sub_terrain, runner;
        public string[] joint_names, body_names, actor_obs_groups;
        public int seed;
        public float physics_dt, step_dt, saturation_effort;

        public int Steps => obs.Rows;

        public static ReferenceRollout Load(string variant) => LoadFile(RepoPaths.ReferenceJson(variant));

        public static ReferenceRollout LoadFile(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"reference rollout {path} not found (recorded on the Linux machine; git pull)");
            return JsonUtility.FromJson<ReferenceRollout>(File.ReadAllText(path));
        }
    }

#pragma warning restore IDE1006, CS0649
}
