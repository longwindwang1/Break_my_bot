using System;
using System.Linq;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// The Go2 articulation as the controller sees it: joints in policy order (robot_interface.yaml joint_names),
    /// base state, torque output, reset. Built and wired by BMB/Setup Go2 (Editor/Go2Setup.cs).
    /// Joint angles are in radians with the URDF sign (Unity's revolute anchor axis is the converted URDF axis).
    /// </summary>
    [DisallowMultipleComponent]
    public sealed class Go2Body : MonoBehaviour
    {
        [SerializeField] ArticulationBody rootBody;
        [Tooltip("Revolute joints in policy order (robot_interface.yaml joint_names)")]
        [SerializeField] ArticulationBody[] joints;
        [SerializeField] string[] jointNames;
        [Tooltip("PhysX joint velocity limits (rad/s), policy order, from the Isaac reference model")]
        [SerializeField] float[] jointVelocityLimits;

        public ArticulationBody Root => rootBody;
        public int NumJoints => joints.Length;
        public string JointName(int i) => jointNames[i];

        /// <summary>Editor setup only.</summary>
        public void Configure(ArticulationBody root, ArticulationBody[] policyJoints, string[] names, float[] velocityLimits)
        {
            rootBody = root;
            joints = policyJoints;
            jointNames = names;
            jointVelocityLimits = velocityLimits;
        }

        void Awake() => ApplyRuntimeSettings();

        /// <summary>
        /// ArticulationBody settings Unity does not serialize (they read 0 on a prefab): velocity caps,
        /// depenetration speed, solver iterations. Applied on Awake; call it yourself in edit-mode code.
        /// </summary>
        public void ApplyRuntimeSettings()
        {
            foreach (var b in rootBody.GetComponentsInChildren<ArticulationBody>(true))
            {
                b.maxDepenetrationVelocity = SimSettings.MaxDepenetrationVelocity;
                b.maxLinearVelocity = SimSettings.MaxLinearVelocity;
                b.maxAngularVelocity = SimSettings.MaxAngularVelocity;
            }
            for (var i = 0; i < joints.Length; i++)
                joints[i].maxJointVelocity = jointVelocityLimits[i];
            rootBody.solverIterations = SimSettings.SolverIterations;
            rootBody.solverVelocityIterations = SimSettings.SolverVelocityIterations;
        }

        /// <summary>Throws if the wired joints do not follow the interface's joint order.</summary>
        public void Validate(RobotInterface itf)
        {
            if (rootBody == null || joints == null || joints.Any(j => j == null))
                throw new InvalidOperationException($"{name}: Go2Body is not wired; rerun BMB/Setup Go2");
            if (!jointNames.SequenceEqual(itf.JointNames))
                throw new InvalidOperationException($"{name}: joint order [{string.Join(", ", jointNames)}] differs from robot_interface.yaml; rerun BMB/Setup Go2");
        }

        // --- state (Unity frame; IsaacFrame converts) -------------------------------------------------------

        public Vector3 BasePosition => rootBody.transform.position;
        public Quaternion BaseRotation => rootBody.transform.rotation;

        /// <summary>Linear velocity of the base centre of mass (base_lin_vel uses the COM velocity).</summary>
        public Vector3 BaseComLinearVelocity => rootBody.linearVelocity;

        public Vector3 BaseAngularVelocity => rootBody.angularVelocity;

        public void ReadJoints(float[] q, float[] qd)
        {
            for (var i = 0; i < joints.Length; i++)
            {
                q[i] = joints[i].jointPosition[0];
                qd[i] = joints[i].jointVelocity[0];
            }
        }

        // --- actuation ------------------------------------------------------------------------------------

        /// <summary>
        /// Joint efforts (N·m), policy order; positive effort increases the joint angle. jointForce persists until
        /// overwritten (it is not cleared after a step), so call this every physics step; Teleport zeroes it.
        /// </summary>
        public void SetJointTorques(float[] tau)
        {
            for (var i = 0; i < joints.Length; i++)
                joints[i].jointForce = new ArticulationReducedSpace(tau[i]);
        }

        // --- reset ----------------------------------------------------------------------------------------

        /// <summary>Places the base and sets joint angles (policy order) with zero velocities.</summary>
        public void Teleport(Vector3 basePosition, Quaternion baseRotation, float[] q)
        {
            rootBody.TeleportRoot(basePosition, baseRotation);
            rootBody.linearVelocity = Vector3.zero;
            rootBody.angularVelocity = Vector3.zero;
            for (var i = 0; i < joints.Length; i++)
            {
                joints[i].jointPosition = new ArticulationReducedSpace(q[i]);
                joints[i].jointVelocity = new ArticulationReducedSpace(0f);
                joints[i].jointForce = new ArticulationReducedSpace(0f);
            }
        }

        // --- contact ---------------------------------------------------------------------------------------

        BoxCollider[] baseBoxes;
        readonly Collider[] overlapHits = new Collider[8];

        /// <summary>
        /// Isaac's base_contact termination (contact force on the base body): true when a box collider of the base
        /// overlaps the given layers, grown by <paramref name="margin"/>. Works in preview scenes without callbacks.
        /// </summary>
        public bool BaseTouches(PhysicsScene physics, int layerMask, float margin = 0.005f)
        {
            baseBoxes ??= rootBody.GetComponentsInChildren<BoxCollider>(true)
                .Where(c => c.GetComponentInParent<ArticulationBody>() == rootBody).ToArray();
            foreach (var box in baseBoxes)
            {
                var t = box.transform;
                var half = Vector3.Scale(box.size, t.lossyScale) * 0.5f + Vector3.one * margin;
                if (physics.OverlapBox(t.TransformPoint(box.center), half, overlapHits, t.rotation, layerMask, QueryTriggerInteraction.Ignore) > 0)
                    return true;
            }
            return false;
        }

        PhysicsMaterial footMaterial;

        /// <summary>
        /// Foot friction like v2's randomize_foot_friction: static μ on the foot colliders, dynamic = ratio · μ
        /// (robot_interface.yaml friction.foot_dynamic_ratio), terrain material multiplied in by the combine mode.
        /// Other links keep the prefab's v1 material. Also the hook for game mechanics such as ice.
        /// </summary>
        public void SetFootFriction(float staticFriction, float dynamicRatio, PhysicsMaterialCombine combine)
        {
            if (footMaterial == null)
                footMaterial = new PhysicsMaterial("Go2_foot_runtime") { hideFlags = HideFlags.DontSave };
            footMaterial.staticFriction = staticFriction;
            footMaterial.dynamicFriction = staticFriction * dynamicRatio;
            footMaterial.frictionCombine = combine;
            footMaterial.bounciness = 0f;
            footMaterial.bounceCombine = PhysicsMaterialCombine.Multiply;
            foreach (var c in rootBody.GetComponentsInChildren<Collider>(true))
                if (c.GetComponentInParent<ArticulationBody>().name.EndsWith("_foot"))
                    c.sharedMaterial = footMaterial;
        }
    }
}
