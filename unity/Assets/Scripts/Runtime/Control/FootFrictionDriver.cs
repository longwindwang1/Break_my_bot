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
        readonly float baseStatic, baseDynamic;
        readonly PhysicsMaterialCombine combine;
        readonly float[] applied = { float.NaN, float.NaN, float.NaN, float.NaN };

        public FrictionZones Zones { get; set; }
        /// <summary>Ground μ under each foot (FL, FR, RL, RR) at the last update.</summary>
        public float[] GroundMu { get; } = new float[4];
        /// <summary>Effective static contact friction of each foot (base × ground).</summary>
        public float EffectiveStatic(int foot) => baseStatic * GroundMu[foot];
        public float BaseStatic => baseStatic;

        /// <param name="footFriction">Static foot μ; ≤ 0 = the v1 material (friction.v1_foot).</param>
        public FootFrictionDriver(Go2Body body, RobotInterface itf, float footFriction)
        {
            this.body = body;
            combine = itf.Friction.UnityCombineMode();
            if (footFriction > 0f)
            {
                baseStatic = footFriction;
                baseDynamic = footFriction * itf.Friction.FootDynamicRatio;
            }
            else
            {
                baseStatic = itf.Friction.V1Foot[0];
                baseDynamic = itf.Friction.V1Foot[1];
            }
            Update();
        }

        public void Update()
        {
            for (var i = 0; i < applied.Length; i++)
            {
                var mu = Zones != null ? Zones.MuAt(body.FootPosition(i)) : 1f;
                GroundMu[i] = mu;
                if (mu == applied[i])
                    continue;
                body.SetFootFriction(i, baseStatic * mu, baseDynamic * mu, combine);
                applied[i] = mu;
            }
        }
    }
}
