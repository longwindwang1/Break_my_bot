namespace BMB
{
    /// <summary>
    /// Unity-only physics settings that have no counterpart in robot_interface.yaml (docs/unity_handoff.md §7).
    /// Timing comes from the yaml; these are the engine knobs around it. Applied by BMB/Apply interface settings.
    /// </summary>
    public static class SimSettings
    {
        /// <summary>Isaac uses 4 position / 0 velocity iterations; Unity's minimum sensible choice is 4 / 1.</summary>
        public const int SolverIterations = 4;
        public const int SolverVelocityIterations = 1;

        /// <summary>Isaac rigid body max_depenetration_velocity (applied per body by the robot setup).</summary>
        public const float MaxDepenetrationVelocity = 1.0f;

        /// <summary>Isaac Lab UNITREE_GO2_CFG rigid_props max_linear/angular_velocity: effectively unlimited.</summary>
        public const float MaxLinearVelocity = 1000f;
        public const float MaxAngularVelocity = 1000f;

        /// <summary>Height-scan rays hit only this layer ("terrain only", §5.6): ground, obstacles, level pieces.</summary>
        public const string TerrainLayer = "Terrain";

        /// <summary>Robot links; Robot×Robot collisions are off (Isaac: self-collision disabled).</summary>
        public const string RobotLayer = "Robot";
    }
}
