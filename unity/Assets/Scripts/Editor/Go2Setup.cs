using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Unity.Robotics.UrdfImporter;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace BMB.Editor
{
    /// <summary>
    /// Builds Assets/Robots/Go2.prefab from the official URDF so that it matches the model Isaac Lab simulated
    /// (docs/unity_handoff.md §7). Idempotent: rerun after the URDF, the yaml or the reference rollouts change.
    ///
    /// 1. Import the URDF with the Unity URDF Importer, then strip every importer component (its Controller would
    ///    drive the joints at runtime).
    /// 2. Keep only the 19 bodies Isaac simulates (reference body_names). The URDF's extra links (motor rotors,
    ///    camera, radar, imu, calf-lower collision links) lose their ArticulationBody: their visuals and colliders
    ///    become part of the parent body and their mass is dropped (Isaac's model has none of it; the importer
    ///    would otherwise give the mass-less links 1 kg each).
    /// 3. Mass, centre of mass and inertia from the reference rollout; joint limits / velocity limits likewise.
    ///    No damping, no joint friction, drives at zero gains (torques come from DCMotor).
    /// 4. Robot layer, v1 friction material (robot_interface.yaml friction), Go2Body wired in policy joint order.
    /// </summary>
    public static class Go2Setup
    {
        public const string UrdfAssetPath = "Assets/Robots/go2_description.urdf";
        public const string PrefabPath = "Assets/Robots/Go2.prefab";
        public const string FrictionMaterialPath = "Assets/Robots/Go2_v1_friction.asset";

        /// <summary>Any reference rollout carries the simulated model; all variants use the same robot.</summary>
        const string ReferenceVariant = "flat";

        [MenuItem("BMB/Setup Go2")]
        public static void Build()
        {
            var itf = RobotInterface.Load();
            var reference = ReferenceRollout.Load(ReferenceVariant);

            // The importer builds into the active scene, so give it a throwaway one: the user's scenes stay clean
            // (a dirty untitled scene makes the test runner block on a save dialog).
            var previous = SceneManager.GetActiveScene();
            if (string.IsNullOrEmpty(previous.path))
                throw new InvalidOperationException("Save or close the untitled scene before running BMB/Setup Go2");
            var temp = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Additive);
            SceneManager.SetActiveScene(temp);
            try
            {
                var robot = Import();
                var report = Configure(robot, itf, reference);
                PrefabUtility.SaveAsPrefabAsset(robot, PrefabPath);
                Debug.Log($"[BMB] Setup Go2 -> {PrefabPath}\n{report}");
            }
            finally
            {
                SceneManager.SetActiveScene(previous);
                EditorSceneManager.CloseScene(temp, true);
            }
        }

        static GameObject Import()
        {
            var settings = ImportSettings.DefaultSettings();
            var it = UrdfRobotExtensions.Create(Path.GetFullPath(UrdfAssetPath), settings, false, false);
            GameObject robot = null;
            while (it.MoveNext())
                if (it.Current != null)
                    robot = it.Current;
            robot ??= SceneManager.GetActiveScene().GetRootGameObjects().FirstOrDefault(g => g.GetComponent<UrdfRobot>() != null);
            if (robot == null)
                throw new InvalidOperationException($"URDF import of {UrdfAssetPath} produced no robot");
            return robot;
        }

        static string Configure(GameObject robot, RobotInterface itf, ReferenceRollout reference)
        {
            var report = new StringBuilder();
            robot.name = "Go2";

            // 1. Strip importer components (dependents before the ArticulationBodies they require).
            var urdfAssembly = typeof(UrdfRobot).Assembly;
            foreach (var c in robot.GetComponentsInChildren<Component>(true).Where(c => c != null && c.GetType().Assembly == urdfAssembly).ToList())
                UnityEngine.Object.DestroyImmediate(c);
            var plugins = robot.transform.Find("Plugins");
            if (plugins != null)
                UnityEngine.Object.DestroyImmediate(plugins.gameObject);

            // 2. Keep the bodies Isaac simulates; deepest first so no body is removed under a kept child.
            var bodyIndex = reference.body_names.Select((n, i) => (n, i)).ToDictionary(p => p.n, p => p.i);
            var removed = new List<string>();
            var removedMass = 0f;
            foreach (var ab in robot.GetComponentsInChildren<ArticulationBody>(true).Reverse())
            {
                if (bodyIndex.ContainsKey(ab.name))
                    continue;
                removed.Add(ab.name);
                removedMass += ab.mass;
                UnityEngine.Object.DestroyImmediate(ab);
            }
            report.AppendLine($"removed {removed.Count} non-Isaac bodies ({removedMass:F3} kg as imported): {string.Join(", ", removed.OrderBy(n => n))}");

            var bodies = robot.GetComponentsInChildren<ArticulationBody>(true);
            var missing = reference.body_names.Except(bodies.Select(b => b.name)).ToList();
            if (missing.Count > 0 || bodies.Length != reference.body_names.Length)
                throw new InvalidOperationException($"bodies after cleanup do not match the reference: missing [{string.Join(", ", missing)}], have {bodies.Length}");

            // 3. Inertial properties and per-body settings.
            float maxComDiff = 0f, maxInertiaDiff = 0f;
            foreach (var ab in bodies)
            {
                var k = bodyIndex[ab.name];
                var com = IsaacFrame.PolarFromIsaac(new Vector3(reference.body_com_pose[k, 0], reference.body_com_pose[k, 1], reference.body_com_pose[k, 2]));
                var (diag, rotation) = Inertia.ToUnity(reference.body_inertia.Row(k));
                maxComDiff = Mathf.Max(maxComDiff, (ab.centerOfMass - com).magnitude);
                maxInertiaDiff = Mathf.Max(maxInertiaDiff, Inertia.MaxAbsDiff(ab.inertiaTensor, ab.inertiaTensorRotation, diag, rotation));

                ab.automaticCenterOfMass = false;
                ab.automaticInertiaTensor = false;
                ab.mass = reference.body_mass.data[k];
                ab.centerOfMass = com;
                ab.inertiaTensor = diag;
                ab.inertiaTensorRotation = rotation;
                ab.linearDamping = 0f;
                ab.angularDamping = 0f;
                ab.jointFriction = itf.Actuator.Friction;
                ab.useGravity = true;
                // Velocity caps, depenetration speed and solver iterations are not serialized by Unity:
                // Go2Body.ApplyRuntimeSettings sets them on Awake.
            }
            report.AppendLine($"inertials from reference: total mass {bodies.Sum(b => b.mass):F3} kg; importer differed by up to {maxComDiff * 1000:F2} mm (COM), {maxInertiaDiff:E1} kg m^2 (inertia)");

            var root = bodies.Single(b => b.isRoot);
            if (root.name != reference.body_names[0])
                throw new InvalidOperationException($"articulation root is {root.name}, expected {reference.body_names[0]}");
            root.immovable = false;

            // Joints in policy order; the URDF names each revolute joint "<link>_joint".
            if (!reference.joint_names.SequenceEqual(itf.JointNames))
                throw new InvalidOperationException("reference joint_names differ from robot_interface.yaml");
            var joints = new ArticulationBody[itf.NumJoints];
            for (var i = 0; i < joints.Length; i++)
            {
                var linkName = itf.JointNames[i].Replace("_joint", "");
                var ab = bodies.SingleOrDefault(b => b.name == linkName)
                         ?? throw new InvalidOperationException($"no body {linkName} for joint {itf.JointNames[i]}");
                if (ab.jointType != ArticulationJointType.RevoluteJoint)
                    throw new InvalidOperationException($"{linkName} is {ab.jointType}, expected a revolute joint");
                var drive = ab.xDrive;
                drive.stiffness = 0f;
                drive.damping = 0f;
                drive.target = 0f;
                drive.targetVelocity = 0f;
                drive.lowerLimit = reference.joint_pos_limits[i, 0] * Mathf.Rad2Deg;
                drive.upperLimit = reference.joint_pos_limits[i, 1] * Mathf.Rad2Deg;
                ab.xDrive = drive;
                ab.twistLock = ArticulationDofLock.LimitedMotion;
                joints[i] = ab;
            }

            // 4. Collision layer and friction.
            var robotLayer = LayerMask.NameToLayer(SimSettings.RobotLayer);
            if (robotLayer < 0)
                throw new InvalidOperationException("Robot layer missing; run BMB/Apply interface settings");
            foreach (var t in robot.GetComponentsInChildren<Transform>(true))
                t.gameObject.layer = robotLayer;
            var friction = FrictionMaterial(itf);
            var colliders = robot.GetComponentsInChildren<Collider>(true);
            foreach (var c in colliders)
            {
                if (c is MeshCollider mc)
                    mc.convex = true;  // articulation bodies need convex colliders
                c.sharedMaterial = friction;
            }
            report.AppendLine($"{colliders.Length} colliders on layer {SimSettings.RobotLayer}, material {friction.name} ({friction.staticFriction}/{friction.dynamicFriction}, {friction.frictionCombine})");

            report.AppendLine(ConvertVisualMaterials(robot));

            var body = robot.AddComponent<Go2Body>();
            body.Configure(root, joints, itf.JointNames.ToArray(), reference.joint_vel_limits.data.ToArray());
            body.Validate(itf);
            return report.ToString();
        }

        public const string VisualMaterialDir = "Assets/Robots/Materials";

        /// <summary>
        /// The .dae meshes embed built-in "Standard" materials, which render magenta under URP. Replace each with a
        /// URP Lit copy (same colour / texture / smoothness) saved next to the importer's own materials.
        /// </summary>
        static string ConvertVisualMaterials(GameObject robot)
        {
            var lit = Shader.Find("Universal Render Pipeline/Lit");
            if (lit == null)
                return "visual materials: URP Lit shader not found, left unchanged";
            var converted = new Dictionary<Material, Material>();
            foreach (var renderer in robot.GetComponentsInChildren<Renderer>(true))
            {
                var materials = renderer.sharedMaterials;
                for (var i = 0; i < materials.Length; i++)
                {
                    var src = materials[i];
                    if (src == null || src.shader == lit)
                        continue;
                    if (!converted.TryGetValue(src, out var dst))
                    {
                        var path = $"{VisualMaterialDir}/{src.name}.mat";
                        dst = AssetDatabase.LoadAssetAtPath<Material>(path);
                        if (dst == null)
                        {
                            dst = new Material(lit);
                            AssetDatabase.CreateAsset(dst, path);
                        }
                        dst.shader = lit;
                        dst.SetColor("_BaseColor", src.HasProperty("_Color") ? src.color : Color.gray);
                        if (src.HasProperty("_MainTex") && src.mainTexture != null)
                            dst.SetTexture("_BaseMap", src.mainTexture);
                        dst.SetFloat("_Smoothness", src.HasProperty("_Glossiness") ? src.GetFloat("_Glossiness") : 0.4f);
                        dst.SetFloat("_Metallic", src.HasProperty("_Metallic") ? src.GetFloat("_Metallic") : 0f);
                        EditorUtility.SetDirty(dst);
                        converted[src] = dst;
                    }
                    materials[i] = dst;
                }
                renderer.sharedMaterials = materials;
            }
            AssetDatabase.SaveAssets();
            return $"visual materials: {converted.Count} built-in Standard materials converted to URP Lit";
        }

        /// <summary>v1 friction on every robot collider (robot_interface.yaml friction.v1_foot, combine mode).</summary>
        static PhysicsMaterial FrictionMaterial(RobotInterface itf)
        {
            var material = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>(FrictionMaterialPath);
            if (material == null)
            {
                material = new PhysicsMaterial("Go2_v1_friction");
                AssetDatabase.CreateAsset(material, FrictionMaterialPath);
            }
            material.staticFriction = itf.Friction.V1Foot[0];
            material.dynamicFriction = itf.Friction.V1Foot[1];
            material.frictionCombine = itf.Friction.UnityCombineMode();
            material.bounciness = 0f;
            material.bounceCombine = PhysicsMaterialCombine.Multiply;
            EditorUtility.SetDirty(material);
            AssetDatabase.SaveAssets();
            return material;
        }
    }

    /// <summary>Inertia tensors from the Isaac link frame to Unity's (principal moments + principal-axes rotation).</summary>
    public static class Inertia
    {
        /// <summary>
        /// I_isaac: 3x3 row-major about the COM in the Isaac link frame. Unity axes are a reflection of Isaac's
        /// (u = M i), and a tensor transforms as M I Mᵀ; then diagonalise.
        /// </summary>
        public static (Vector3 diagonal, Quaternion rotation) ToUnity(float[] inertiaIsaac)
        {
            var I = new double[3, 3];
            for (var r = 0; r < 3; r++)
                for (var c = 0; c < 3; c++)
                    I[r, c] = inertiaIsaac[3 * r + c];
            // M rows: u.x = -i.y, u.y = i.z, u.z = i.x
            int[] src = { 1, 2, 0 };
            double[] sign = { -1, 1, 1 };
            var U = new double[3, 3];
            for (var r = 0; r < 3; r++)
                for (var c = 0; c < 3; c++)
                    U[r, c] = sign[r] * sign[c] * I[src[r], src[c]];

            var (values, vectors) = JacobiEigen(U);
            // Proper rotation: columns e0, e1, e2 with det +1.
            var e = new Vector3[3];
            for (var j = 0; j < 3; j++)
                e[j] = new Vector3((float)vectors[0, j], (float)vectors[1, j], (float)vectors[2, j]);
            if (Vector3.Dot(Vector3.Cross(e[0], e[1]), e[2]) < 0)
                e[0] = -e[0];
            var rotation = Quaternion.LookRotation(e[2], e[1]);  // columns (e1 × e2, e1, e2)
            return (new Vector3((float)values[0], (float)values[1], (float)values[2]), rotation);
        }

        /// <summary>R diag(d) Rᵀ as a 3x3 row-major array.</summary>
        public static float[] Tensor(Vector3 diagonal, Quaternion rotation)
        {
            var R = Matrix4x4.Rotate(rotation);
            var T = new float[9];
            for (var r = 0; r < 3; r++)
                for (var c = 0; c < 3; c++)
                    T[3 * r + c] = R[r, 0] * diagonal.x * R[c, 0] + R[r, 1] * diagonal.y * R[c, 1] + R[r, 2] * diagonal.z * R[c, 2];
            return T;
        }

        public static float MaxAbsDiff(Vector3 diagA, Quaternion rotA, Vector3 diagB, Quaternion rotB)
        {
            var a = Tensor(diagA, rotA);
            var b = Tensor(diagB, rotB);
            return a.Zip(b, (x, y) => Mathf.Abs(x - y)).Max();
        }

        /// <summary>Cyclic Jacobi for a symmetric 3x3 matrix; eigenvectors are the columns.</summary>
        static (double[] values, double[,] vectors) JacobiEigen(double[,] a)
        {
            var A = (double[,])a.Clone();
            var V = new double[3, 3] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
            for (var sweep = 0; sweep < 50; sweep++)
            {
                var off = Math.Abs(A[0, 1]) + Math.Abs(A[0, 2]) + Math.Abs(A[1, 2]);
                if (off < 1e-18)
                    break;
                for (var p = 0; p < 2; p++)
                    for (var q = p + 1; q < 3; q++)
                    {
                        if (Math.Abs(A[p, q]) < 1e-30)
                            continue;
                        var theta = (A[q, q] - A[p, p]) / (2 * A[p, q]);
                        var t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1));
                        if (theta == 0) t = 1;
                        var c = 1 / Math.Sqrt(t * t + 1);
                        var s = t * c;
                        for (var k = 0; k < 3; k++)
                        {
                            var akp = A[k, p];
                            var akq = A[k, q];
                            A[k, p] = c * akp - s * akq;
                            A[k, q] = s * akp + c * akq;
                        }
                        for (var k = 0; k < 3; k++)
                        {
                            var apk = A[p, k];
                            var aqk = A[q, k];
                            A[p, k] = c * apk - s * aqk;
                            A[q, k] = s * apk + c * aqk;
                        }
                        for (var k = 0; k < 3; k++)
                        {
                            var vkp = V[k, p];
                            var vkq = V[k, q];
                            V[k, p] = c * vkp - s * vkq;
                            V[k, q] = s * vkp + c * vkq;
                        }
                    }
            }
            return (new[] { A[0, 0], A[1, 1], A[2, 2] }, V);
        }
    }
}
