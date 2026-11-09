using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.SceneManagement;

namespace BMB
{
    /// <summary>
    /// A visual-only copy of a Go2 (no articulation, no colliders) posed by forward kinematics: base pose plus joint
    /// angles in policy order. The kinematics use the source articulation's joint frames, the same composition Unity
    /// uses for an articulation: child = parent · parentAnchor · Rx(q) · anchor⁻¹ (fixed joints: q = 0).
    /// </summary>
    public sealed class GhostRobot : MonoBehaviour
    {
        struct LinkFrame
        {
            public Transform Transform;
            public int Parent;  // index into links; -1 for the base
            public Vector3 ParentAnchorPosition, AnchorPosition;
            public Quaternion ParentAnchorRotation, AnchorRotation;
            public int Joint;   // policy joint index, -1 for fixed joints
        }

        LinkFrame[] links;
        Vector3[] worldPos;
        Quaternion[] worldRot;
        Renderer[] renderers;

        /// <summary>Link transforms by name (base, FL_hip, ..., RR_foot).</summary>
        public Transform Link(string linkName)
        {
            foreach (var l in links)
                if (l.Transform.name == linkName)
                    return l.Transform;
            return null;
        }

        public IReadOnlyList<Renderer> Renderers => renderers;

        /// <summary>Builds the ghost next to <paramref name="source"/> (same scene), all renderers using <paramref name="material"/>.</summary>
        public static GhostRobot Create(Go2Body source, Material material, string name = "Go2Ghost")
        {
            var container = new GameObject("GhostBuild");
            container.SetActive(false);  // the copy's articulation never enters the physics scene
            SceneManager.MoveGameObjectToScene(container, source.gameObject.scene);
            var clone = Instantiate(source.gameObject, container.transform);
            clone.name = name;

            var bodies = source.Root.GetComponentsInChildren<ArticulationBody>(true);  // parents before children
            var jointOf = new Dictionary<ArticulationBody, int>();
            for (var i = 0; i < source.NumJoints; i++)
                jointOf[source.Joint(i)] = i;
            var links = new LinkFrame[bodies.Length];
            for (var k = 0; k < bodies.Length; k++)
            {
                var b = bodies[k];
                var parent = k == 0 ? null : b.transform.parent.GetComponentInParent<ArticulationBody>(true);
                links[k] = new LinkFrame
                {
                    Transform = Find(clone.transform, source.transform, b.transform),
                    Parent = parent == null ? -1 : System.Array.IndexOf(bodies, parent),
                    ParentAnchorPosition = b.parentAnchorPosition,
                    ParentAnchorRotation = b.parentAnchorRotation,
                    AnchorPosition = b.anchorPosition,
                    AnchorRotation = b.anchorRotation,
                    Joint = b.jointType == ArticulationJointType.RevoluteJoint && jointOf.TryGetValue(b, out var j) ? j : -1,
                };
            }

            // Strip everything but transforms and renderers: scripts first (they may require the bodies), then
            // colliders, then articulation bodies children first.
            foreach (var mb in clone.GetComponentsInChildren<MonoBehaviour>(true))
                DestroyImmediate(mb);
            foreach (var c in clone.GetComponentsInChildren<Collider>(true))
                DestroyImmediate(c);
            var cloneBodies = clone.GetComponentsInChildren<ArticulationBody>(true);
            for (var k = cloneBodies.Length - 1; k >= 0; k--)
                DestroyImmediate(cloneBodies[k]);
            foreach (var t in clone.GetComponentsInChildren<Transform>(true))
                t.gameObject.layer = 0;  // Default: invisible to the height scan and the tools' robot picking

            var ghost = clone.AddComponent<GhostRobot>();
            ghost.links = links;
            ghost.worldPos = new Vector3[links.Length];
            ghost.worldRot = new Quaternion[links.Length];
            ghost.renderers = clone.GetComponentsInChildren<Renderer>(true);
            ghost.SetMaterial(material);
            clone.transform.SetParent(null, true);
            DestroyImmediate(container);
            return ghost;
        }

        static Transform Find(Transform cloneRoot, Transform sourceRoot, Transform sourceT)
        {
            var path = new List<string>();
            for (var t = sourceT; t != sourceRoot; t = t.parent)
                path.Insert(0, t.name);
            return path.Count == 0 ? cloneRoot : cloneRoot.Find(string.Join("/", path));
        }

        public void SetMaterial(Material material)
        {
            foreach (var r in renderers)
            {
                var mats = new Material[r.sharedMaterials.Length];
                for (var i = 0; i < mats.Length; i++)
                    mats[i] = material;
                r.sharedMaterials = mats;
                r.shadowCastingMode = ShadowCastingMode.Off;
                r.receiveShadows = false;
            }
        }

        /// <summary>Base pose (Unity frame) and joint angles (rad, policy order, absolute).</summary>
        public void SetPose(Vector3 basePosition, Quaternion baseRotation, IReadOnlyList<float> jointPos)
        {
            for (var k = 0; k < links.Length; k++)
            {
                var l = links[k];
                if (l.Parent < 0)
                {
                    worldPos[k] = basePosition;
                    worldRot[k] = baseRotation;
                }
                else
                {
                    var (pp, pr) = (worldPos[l.Parent], worldRot[l.Parent]);
                    var joint = l.Joint >= 0 ? Quaternion.AngleAxis(jointPos[l.Joint] * Mathf.Rad2Deg, Vector3.right) : Quaternion.identity;
                    var r = pr * l.ParentAnchorRotation * joint * Quaternion.Inverse(l.AnchorRotation);
                    worldRot[k] = r;
                    worldPos[k] = pp + pr * l.ParentAnchorPosition - r * l.AnchorPosition;
                }
                l.Transform.SetPositionAndRotation(worldPos[k], worldRot[k]);
            }
        }
    }
}
