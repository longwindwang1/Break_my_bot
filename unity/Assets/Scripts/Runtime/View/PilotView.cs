using UnityEngine;

namespace BMB
{
    /// <summary>
    /// What the AI pilot is doing, drawn in the world: the challenge goal (green ring + beacon), the current walk-to
    /// target (small white ring), and the world model's look ahead along the running order (amber, red when it predicts
    /// a fall) or the last imagined order (orange, for a few seconds after an imagine call or a veto).
    /// Play mode only (line renderers are scene objects).
    /// </summary>
    public sealed class PilotView : MonoBehaviour
    {
        public static readonly Color GoalColour = new Color(0.22f, 1f, 0.53f, 0.95f);
        public static readonly Color TargetColour = new Color(0.9f, 0.95f, 1f, 0.8f);
        public static readonly Color LookaheadColour = new Color(1f, 0.72f, 0.2f, 0.85f);
        public static readonly Color ImaginedColour = new Color(1f, 0.45f, 0.1f, 0.95f);
        public static readonly Color RiskColour = new Color(1f, 0.2f, 0.25f, 0.95f);
        const float ImaginedShowSeconds = 4f;
        const int RingSegments = 48;

        [SerializeField] ClaudePilot pilot;
        [SerializeField] PilotChallenge challenge;

        LineRenderer goalRing, goalBeam, targetRing, path;
        Vector3[] imaginedPath = System.Array.Empty<Vector3>();
        bool imaginedRisky;
        float imaginedUntil;

        public void Configure(ClaudePilot owner, PilotChallenge round)
        {
            pilot = owner;
            challenge = round;
        }

        void OnEnable()
        {
            if (pilot != null)
                pilot.Imagined += OnImagined;
        }

        void OnDisable()
        {
            if (pilot != null)
                pilot.Imagined -= OnImagined;
        }

        void OnImagined(ClaudePilot p, PilotImagined imagined)
        {
            imaginedPath = (Vector3[])p.Imagination.LastPath.Clone();
            imaginedRisky = p.Veto && ClaudePilot.Risk(imagined.max_fall_prob) > p.VetoFallProb;
            imaginedUntil = Time.unscaledTime + ImaginedShowSeconds;
        }

        LineRenderer Line(string name, float width, bool loop, int count)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            var lr = go.AddComponent<LineRenderer>();
            lr.useWorldSpace = true;
            lr.loop = loop;
            lr.widthMultiplier = width;
            lr.numCapVertices = 2;
            lr.positionCount = count;
            lr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            lr.receiveShadows = false;
            lr.sharedMaterial = new Material(Shader.Find("Sprites/Default"));
            go.SetActive(false);
            return lr;
        }

        void EnsureLines()
        {
            if (goalRing != null)
                return;
            goalRing = Line("PilotGoalRing", 0.05f, true, RingSegments);
            goalBeam = Line("PilotGoalBeam", 0.06f, false, 2);
            targetRing = Line("PilotTargetRing", 0.02f, true, RingSegments);
            path = Line("PilotPath", 0.025f, false, 0);
        }

        void LateUpdate()
        {
            if (!Application.isPlaying || pilot == null || pilot.Controller == null || !pilot.Controller.Initialized)
                return;
            EnsureLines();
            var c = pilot.Controller;

            // Goal: shown while a round runs (and after it, until the next one) or whenever the pilot has a goal.
            if (pilot.Goal is Vector2 goal)
            {
                var at = Ground(c, goal);
                var pulse = 1f + 0.08f * Mathf.Sin(Time.unscaledTime * 4f);
                Ring(goalRing, at, PilotChallenge.GoalRadius * pulse, GoalColour);
                var reached = challenge != null && !challenge.Running && challenge.LastResult != null && challenge.LastResult.success;
                goalBeam.gameObject.SetActive(true);
                goalBeam.startColor = GoalColour;
                goalBeam.endColor = new Color(GoalColour.r, GoalColour.g, GoalColour.b, 0f);
                goalBeam.SetPosition(0, at);
                goalBeam.SetPosition(1, at + Vector3.up * (reached ? 3f : 1.5f));
            }
            else
            {
                goalRing.gameObject.SetActive(false);
                goalBeam.gameObject.SetActive(false);
            }

            // Current walk-to target.
            var o = pilot.Order;
            if (pilot.Engaged && o != null && o.Running && o.Kind == PilotOrderKind.WalkTo)
                Ring(targetRing, Ground(c, o.Target), PilotDrive.ArriveRadius, TargetColour);
            else
                targetRing.gameObject.SetActive(false);

            // Imagined order (recent imagine / veto) has priority over the running lookahead.
            if (Time.unscaledTime < imaginedUntil && imaginedPath.Length > 1)
                Path(imaginedPath, imaginedRisky ? RiskColour : ImaginedColour);
            else if (pilot.Engaged && o != null && o.Running && pilot.Veto && pilot.LookaheadPath.Length > 1)
                Path(pilot.LookaheadPath, pilot.LookaheadRisk > pilot.VetoFallProb * 0.6f ? RiskColour : LookaheadColour);
            else
                path.gameObject.SetActive(false);
        }

        static Vector3 Ground(Go2Controller c, Vector2 isaac)
        {
            var u = IsaacFrame.PolarFromIsaac(new Vector3(isaac.x, isaac.y, 0f));
            u.y = c.GroundHeightAt(u) + 0.03f;
            return u;
        }

        static void Ring(LineRenderer line, Vector3 centre, float radius, Color colour)
        {
            line.gameObject.SetActive(true);
            line.startColor = line.endColor = colour;
            for (var i = 0; i < RingSegments; i++)
            {
                var a = i * Mathf.PI * 2f / RingSegments;
                line.SetPosition(i, centre + new Vector3(Mathf.Cos(a) * radius, 0f, Mathf.Sin(a) * radius));
            }
        }

        void Path(Vector3[] points, Color colour)
        {
            path.gameObject.SetActive(true);
            path.positionCount = points.Length;
            path.SetPositions(points);
            path.startColor = colour;
            path.endColor = new Color(colour.r, colour.g, colour.b, 0.25f);
        }
    }
}
