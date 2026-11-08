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

        // --- feet ------------------------------------------------------------------------------------------

        /// <summary>Foot bodies in this order (index used by the per-foot methods).</summary>
        public static readonly string[] FootNames = { "FL_foot", "FR_foot", "RL_foot", "RR_foot" };

        Transform[] feet;
        ArticulationBody[] footBodies;
        Collider[][] footColliders;
        PhysicsMaterial[] footMaterials;

        void FindFeet()
        {
            if (feet != null)
                return;
            var bodies = rootBody.GetComponentsInChildren<ArticulationBody>(true);
            var colliders = rootBody.GetComponentsInChildren<Collider>(true);
            feet = new Transform[FootNames.Length];
            footBodies = new ArticulationBody[FootNames.Length];
            footColliders = new Collider[FootNames.Length][];
            footMaterials = new PhysicsMaterial[FootNames.Length];
            for (var i = 0; i < FootNames.Length; i++)
            {
                var foot = bodies.Single(b => b.name == FootNames[i]);
                feet[i] = foot.transform;
                footBodies[i] = foot;
                footColliders[i] = colliders.Where(c => c.GetComponentInParent<ArticulationBody>() == foot).ToArray();
                footMaterials[i] = new PhysicsMaterial($"Go2_{FootNames[i]}_runtime") { hideFlags = HideFlags.DontSave, bounciness = 0f, bounceCombine = PhysicsMaterialCombine.Multiply };
            }
        }

        /// <summary>World position of a foot body (Unity frame).</summary>
        public Vector3 FootPosition(int foot)
        {
            FindFeet();
            return feet[foot].position;
        }

        /// <summary>World linear velocity of a foot body (Unity frame).</summary>
        public Vector3 FootVelocity(int foot)
        {
            FindFeet();
            return footBodies[foot].linearVelocity;
        }

        /// <summary>
        /// A foot touches the given layers: its collider grown by <paramref name="margin"/> overlaps them (the
        /// evaluation's stand-in for Isaac's "contact force > 1 N", without contact callbacks).
        /// </summary>
        public bool FootTouches(int foot, PhysicsScene physics, int layerMask, float margin = 0.003f)
        {
            FindFeet();
            foreach (var c in footColliders[foot])
            {
                if (c is SphereCollider s)
                {
                    var t = s.transform;
                    var r = s.radius * Mathf.Max(Mathf.Abs(t.lossyScale.x), Mathf.Abs(t.lossyScale.y), Mathf.Abs(t.lossyScale.z));
                    if (physics.OverlapSphere(t.TransformPoint(s.center), r + margin, overlapHits, layerMask, QueryTriggerInteraction.Ignore) > 0)
                        return true;
                }
                else
                {
                    var b = c.bounds;
                    if (physics.OverlapBox(b.center, b.extents + Vector3.one * margin, overlapHits, Quaternion.identity, layerMask, QueryTriggerInteraction.Ignore) > 0)
                        return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Friction of one foot (v2's randomize_foot_friction writes the foot colliders; the terrain material, μ 1.0
        /// with the same combine mode, multiplies in). Other links keep the prefab's v1 material.
        /// </summary>
        public void SetFootFriction(int foot, float staticFriction, float dynamicFriction, PhysicsMaterialCombine combine)
        {
            FindFeet();
            var m = footMaterials[foot];
            m.staticFriction = staticFriction;
            m.dynamicFriction = dynamicFriction;
            m.frictionCombine = combine;
            foreach (var c in footColliders[foot])
                c.sharedMaterial = m;
        }

        /// <summary>All feet: static μ, dynamic = ratio · μ (robot_interface.yaml friction.foot_dynamic_ratio).</summary>
        public void SetFootFriction(float staticFriction, float dynamicRatio, PhysicsMaterialCombine combine)
        {
            for (var i = 0; i < FootNames.Length; i++)
                SetFootFriction(i, staticFriction, staticFriction * dynamicRatio, combine);
        }

        public float FootStaticFriction(int foot)
        {
            FindFeet();
            return footColliders[foot][0].sharedMaterial.staticFriction;
        }

        // --- payload ---------------------------------------------------------------------------------------

        bool nominalStored;
        float nominalMass;
        Vector3 nominalCom, nominalInertia;

        /// <summary>Mass added to the base (kg) and base COM offset in Isaac base axes (m), as last set.</summary>
        public float AddedMass { get; private set; }
        public Vector3 ComOffset { get; private set; }

        public float TotalMass => rootBody.GetComponentsInChildren<ArticulationBody>(true).Sum(b => b.mass);
        public float NominalBaseMass { get { StoreNominal(); return nominalMass; } }
        public Vector3 NominalBaseCom { get { StoreNominal(); return nominalCom; } }

        void StoreNominal()
        {
            if (nominalStored)
                return;
            nominalMass = rootBody.mass;
            nominalCom = rootBody.centerOfMass;
            nominalInertia = rootBody.inertiaTensor;
            nominalStored = true;
        }

        /// <summary>
        /// Payload like v2's randomize_base_mass_cached / randomize_base_com_cached: base mass = nominal + added, inertia
        /// scaled by the mass ratio, COM = nominal + offset (Isaac base axes: x forward, y left, z up). (0, 0) restores.
        /// </summary>
        public void SetPayload(float addedMass, Vector3 comOffsetIsaac)
        {
            StoreNominal();
            rootBody.mass = nominalMass + addedMass;
            rootBody.inertiaTensor = nominalInertia * (rootBody.mass / nominalMass);
            rootBody.centerOfMass = nominalCom + IsaacFrame.PolarFromIsaac(comOffsetIsaac);
            AddedMass = addedMass;
            ComOffset = comOffsetIsaac;
        }
    }
}
