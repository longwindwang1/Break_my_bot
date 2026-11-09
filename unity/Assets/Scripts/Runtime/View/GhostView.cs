using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// The world model's "blue ghost": a translucent Go2 a little ahead of the robot, where the world model imagines it
    /// will be after <see cref="HorizonSeconds"/> (policy in the loop, see <see cref="WorldModelDreamer"/>).
    /// - A new dream starts right after a policy step once the previous one is shown; dreams advance in Update under a
    ///   time budget, and the shown pose eases towards each finished dream.
    /// - Colour: blue; towards red as the world model's P(fall within 0.5 s) for the real robot rises; a magenta
    ///   flash on every surprisal alarm. The ghost fades where it overlaps the robot.
    /// - A thin line traces the imagined base path. V toggles the ghost.
    /// </summary>
    [DefaultExecutionOrder(100)]
    public sealed class GhostView : MonoBehaviour
    {
        [SerializeField] Go2Controller controller;
        [Tooltip("Translucent material for the ghost (Assets/Game/Ghost.mat); its colour is driven at runtime")]
        [SerializeField] Material ghostMaterial;
        [SerializeField] bool show = true;
        [Tooltip("How far ahead the ghost is imagined (s)")]
        [Range(0.1f, 1f)] [SerializeField] float horizonSeconds = 0.5f;
        [Tooltip("Imagination time per frame (ms); a 0.5 s dream costs about 25 ms in total")]
        [SerializeField] float budgetMs = 4f;
        [Tooltip("Time constant (s) of the eased ghost pose")]
        [SerializeField] float smoothing = 0.08f;

        public static readonly Color Blue = new Color(0.25f, 0.66f, 1f);
        public static readonly Color Risk = new Color(1f, 0.23f, 0.36f);
        public static readonly Color Flash = new Color(1f, 0.17f, 0.84f);
        const float MaxAlpha = 0.42f, MinAlpha = 0.07f;

        WorldModelDreamer dreamer;
        GhostRobot ghost;
        Material material;
        LineRenderer line;
        WorldModelMonitor subscribedMonitor;
        Vector3 shownPosition;
        Quaternion shownRotation;
        float[] shownJoints;
        bool hasShown, startNext;
        float flash;

        public Go2Controller Controller => controller;
        public GhostRobot Ghost => ghost;
        public WorldModelDreamer Dreamer => dreamer;

        public bool Show
        {
            get => show;
            set => show = value;
        }

        public float HorizonSeconds
        {
            get => horizonSeconds;
            set => horizonSeconds = Mathf.Clamp(value, 0.1f, 1f);
        }

        /// <summary>The current tint (before alpha), for the HUD.</summary>
        public Color Tint { get; private set; } = Blue;

        public void Configure(Go2Controller robot, Material material = null)
        {
            controller = robot;
            ghostMaterial = material;
        }

        void OnDisable() => Teardown();

        /// <summary>Removes the ghost and stops listening; <see cref="EnsureCreated"/> builds it again.</summary>
        public void Teardown()
        {
            Unsubscribe();
            if (ghost != null)
                DestroyImmediate(ghost.gameObject);
            if (material != null)
                DestroyImmediate(material);
            if (line != null)
            {
                DestroyImmediate(line.sharedMaterial);
                DestroyImmediate(line.gameObject);
            }
            (ghost, material, line, hasShown) = (null, null, null, false);
        }

        /// <summary>Creates the dreamer and the ghost once the robot's world model runs. Returns false until then.</summary>
        public bool EnsureCreated()
        {
            if (dreamer != null)
                return true;
            if (controller == null || !controller.Initialized || controller.WorldModel == null)
                return false;
            dreamer = new WorldModelDreamer(controller);
            material = ghostMaterial != null ? new Material(ghostMaterial) : new Material(Shader.Find("Sprites/Default"));
            material.name = "Go2Ghost (runtime)";
            ghost = GhostRobot.Create(controller.Body, material);
            shownJoints = new float[controller.Interface.NumJoints];
            controller.PolicyStepped += OnPolicyStepped;
            controller.WasReset += OnReset;
            if (Application.isPlaying)
                line = CreateLine();
            startNext = true;
            return true;
        }

        void Unsubscribe()
        {
            if (controller != null)
            {
                controller.PolicyStepped -= OnPolicyStepped;
                controller.WasReset -= OnReset;
            }
            if (subscribedMonitor != null)
                subscribedMonitor.Alarmed -= OnAlarm;
            subscribedMonitor = null;
            dreamer = null;
        }

        void OnPolicyStepped(Go2Controller c)
        {
            if (subscribedMonitor != c.WorldModel)
            {
                if (subscribedMonitor != null)
                    subscribedMonitor.Alarmed -= OnAlarm;
                subscribedMonitor = c.WorldModel;
                if (subscribedMonitor != null)
                    subscribedMonitor.Alarmed += OnAlarm;
            }
            if (!show || !startNext || dreamer.Busy)
                return;
            dreamer.Horizon = Mathf.Max(1, Mathf.RoundToInt(horizonSeconds / c.Interface.Timing.PolicyDt));
            if (dreamer.Begin())
                startNext = false;
        }

        void OnReset(Go2Controller c)
        {
            dreamer?.Cancel();
            hasShown = false;
            startNext = true;
        }

        void OnAlarm(WorldModelMonitor m) => flash = 1f;

        void Update()
        {
            if (Keyboard.current != null && Keyboard.current.vKey.wasPressedThisFrame)
                show = !show;
            Tick(Time.deltaTime);
        }

        /// <summary>Advances the imagination and the shown pose (Update calls it; tests call it directly).</summary>
        public void Tick(float dt)
        {
            if (!EnsureCreated())
                return;
            var visible = show && hasShown;
            if (ghost.gameObject.activeSelf != visible)
                ghost.gameObject.SetActive(visible);
            if (line != null)
                line.enabled = visible;
            if (!show)
            {
                dreamer.Cancel();
                startNext = true;
                hasShown = false;
                return;
            }

            if (dreamer.Advance(budgetMs))
            {
                startNext = true;
                if (!hasShown)
                {
                    shownPosition = dreamer.GhostPosition;
                    shownRotation = dreamer.GhostRotation;
                    System.Array.Copy(dreamer.GhostJointPos, shownJoints, shownJoints.Length);
                    hasShown = true;
                }
                if (line != null)
                {
                    line.positionCount = dreamer.Path.Length;
                    line.SetPositions(dreamer.Path);
                }
            }
            if (!hasShown)
                return;

            var a = smoothing > 0f ? 1f - Mathf.Exp(-dt / smoothing) : 1f;
            shownPosition = Vector3.Lerp(shownPosition, dreamer.GhostPosition, a);
            shownRotation = Quaternion.Slerp(shownRotation, dreamer.GhostRotation, a);
            for (var i = 0; i < shownJoints.Length; i++)
                shownJoints[i] = Mathf.Lerp(shownJoints[i], dreamer.GhostJointPos[i], a);
            ghost.SetPose(shownPosition, shownRotation, shownJoints);

            // colour: fall risk of the real robot (h = 25 steps), alarm flash, fade where it overlaps the robot
            var m = controller.WorldModel;
            var risk = m != null && m.HasPrediction ? Mathf.Clamp01(m.Prediction.FallProb[Mathf.Min(1, m.Prediction.FallProb.Length - 1)] / 0.5f) : 0f;
            var tint = Color.Lerp(Blue, Risk, risk);
            tint = Color.Lerp(tint, Flash, flash);
            flash = Mathf.Max(0f, flash - dt / 0.5f);
            Tint = tint;
            var apart = Vector3.Distance(shownPosition, controller.Body.BasePosition);
            var alpha = Mathf.Lerp(MinAlpha, MaxAlpha, Mathf.SmoothStep(0f, 1f, Mathf.InverseLerp(0.03f, 0.2f, apart)));
            SetColour(material, new Color(tint.r, tint.g, tint.b, alpha));
            if (line != null)
                line.startColor = line.endColor = new Color(tint.r, tint.g, tint.b, 0.55f);
        }

        static void SetColour(Material m, Color c)
        {
            if (m.HasProperty("_BaseColor"))
                m.SetColor("_BaseColor", c);
            if (m.HasProperty("_Color"))
                m.SetColor("_Color", c);
            if (m.HasProperty("_EmissionColor"))
                m.SetColor("_EmissionColor", new Color(c.r, c.g, c.b) * 0.9f);
        }

        LineRenderer CreateLine()
        {
            var go = new GameObject("GhostPath");
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true;
            lr.widthMultiplier = 0.012f;
            lr.numCapVertices = 2;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            lr.material = new Material(Shader.Find("Sprites/Default"));
            lr.positionCount = 0;
            return lr;
        }
    }
}
