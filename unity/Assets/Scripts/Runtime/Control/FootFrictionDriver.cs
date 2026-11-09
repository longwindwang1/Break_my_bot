using UnityEngine;

namespace BMB
{
    /// <summary>
    /// Per-foot friction from the ground under each foot, the way v2 was trained (randomize_foot_friction writes the
    /// foot material; the terrain stays μ 1.0 with multiply combine). Every physics step: ground μ at each foot from
    /// <see cref="FrictionZones"/> (1.0 without zones), foot material = foot base μ × ground μ, so the effective
    /// contact friction is base × ground. Only changed values are written.
    /// </summary>
    public sealed class FootFrictionDriver
    {
        readonly Go2Body body;
        readonly RobotInterface itf;
        readonly float[] baseStatic = new float[4], baseDynamic = new float[4];
        readonly PhysicsMaterialCombine combine;
        readonly float[] applied = { float.NaN, float.NaN, float.NaN, float.NaN };

        public FrictionZones Zones { get; set; }
        /// <summary>Ground μ under each foot (FL, FR, RL, RR) at the last update.</summary>
        public float[] GroundMu { get; } = new float[4];
        /// <summary>Effective static contact friction of each foot (base × ground).</summary>
        public float EffectiveStatic(int foot) => baseStatic[foot] * GroundMu[foot];
        /// <summary>Static base μ of a foot (before the ground's μ multiplies in).</summary>
        public float BaseStatic(int foot) => baseStatic[foot];

        /// <param name="footFriction">Static foot μ; ≤ 0 = the v1 material (friction.v1_foot).</param>
        public FootFrictionDriver(Go2Body body, RobotInterface itf, float footFriction)
        {
            this.body = body;
            this.itf = itf;
            combine = itf.Friction.UnityCombineMode();
            SetFootMu(footFriction);
        }

        /// <summary>
        /// Changes the base μ of every foot mid-run (like Isaac's randomize_foot_friction event, e.g. the eval's
        /// mu_drop); ≤ 0 = the v1 material. Applied at once.
        /// </summary>
        public void SetFootMu(float footFriction)
        {
            for (var i = 0; i < baseStatic.Length; i++)
                Set(i, footFriction);
            Update();
        }

        /// <summary>One foot only (FL, FR, RL, RR = 0..3), e.g. a single icy foot.</summary>
        public void SetFootMu(int foot, float footFriction)
        {
            Set(foot, footFriction);
            Update();
        }

        void Set(int i, float footFriction)
        {
            if (footFriction > 0f)
            {
                baseStatic[i] = footFriction;
                baseDynamic[i] = footFriction * itf.Friction.FootDynamicRatio;
            }
            else
            {
                baseStatic[i] = itf.Friction.V1Foot[0];
                baseDynamic[i] = itf.Friction.V1Foot[1];
            }
            applied[i] = float.NaN;
        }

        public void Update()
        {
            for (var i = 0; i < applied.Length; i++)
            {
                var mu = Zones != null ? Zones.MuAt(body.FootPosition(i)) : 1f;
                GroundMu[i] = mu;
                if (mu == applied[i])
                    continue;
                body.SetFootFriction(i, baseStatic[i] * mu, baseDynamic[i] * mu, combine);
                applied[i] = mu;
            }
        }
    }
}
