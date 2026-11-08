using UnityEngine;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// Top view of the Go2 (forward = up): neon body outline, legs, and the four feet as discs coloured by the effective
    /// friction under each foot (ice blue → white → orange), with the μ value next to each foot.
    /// </summary>
    public sealed class FootDiagram : VisualElement
    {
        static readonly CustomStyleProperty<Color> LineColorProperty = new CustomStyleProperty<Color>("--line-color");

        // Foot positions in the unit square (FL, FR, RL, RR; forward = up, left = left).
        static readonly Vector2[] Feet = { new Vector2(0.3f, 0.2f), new Vector2(0.7f, 0.2f), new Vector2(0.3f, 0.8f), new Vector2(0.7f, 0.8f) };

        readonly float[] mu = { 1f, 1f, 1f, 1f };
        readonly Label[] labels = new Label[4];
        Color line = new Color(0f, 0.9f, 1f, 0.8f);

        public FootDiagram()
        {
            AddToClassList("foot-diagram");
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
            for (var i = 0; i < 4; i++)
            {
                labels[i] = new Label { pickingMode = PickingMode.Ignore };
                labels[i].AddToClassList("foot-mu");
                labels[i].AddToClassList("num");
                labels[i].AddToClassList(i % 2 == 0 ? "left" : "right");
                labels[i].style.top = Length.Percent(Feet[i].y * 100f);
                Add(labels[i]);
            }
            RegisterCallback<CustomStyleResolvedEvent>(e =>
            {
                if (e.customStyle.TryGetValue(LineColorProperty, out var c))
                {
                    line = c;
                    MarkDirtyRepaint();
                }
            });
        }

        public void SetFriction(int foot, float value)
        {
            if (Mathf.Abs(mu[foot] - value) < 1e-3f && labels[foot].text.Length > 0)
                return;
            mu[foot] = value;
            labels[foot].text = value.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture);
            labels[foot].EnableInClassList("ice", value < 0.45f);
            MarkDirtyRepaint();
        }

        void Draw(MeshGenerationContext ctx)
        {
            var r = contentRect;
            if (r.width < 4f || r.height < 4f)
                return;
            var p = ctx.painter2D;
            Vector2 P(float x, float y) => new Vector2(r.x + x * r.width, r.y + y * r.height);

            p.lineJoin = LineJoin.Round;
            p.lineCap = LineCap.Round;
            foreach (var (w, a) in new[] { (6f, 0.12f), (3f, 0.35f), (1.4f, 1f) })
            {
                p.strokeColor = new Color(line.r, line.g, line.b, line.a * a);
                p.lineWidth = w;
                // Body and head (forward = up).
                p.BeginPath();
                p.MoveTo(P(0.38f, 0.16f)); p.LineTo(P(0.62f, 0.16f)); p.LineTo(P(0.62f, 0.84f)); p.LineTo(P(0.38f, 0.84f));
                p.ClosePath();
                p.Stroke();
                p.BeginPath();
                p.MoveTo(P(0.44f, 0.16f)); p.LineTo(P(0.5f, 0.07f)); p.LineTo(P(0.56f, 0.16f));
                p.Stroke();
                // Legs from the hips to the feet.
                foreach (var (hip, foot) in new[] { (P(0.38f, 0.22f), Feet[0]), (P(0.62f, 0.22f), Feet[1]), (P(0.38f, 0.78f), Feet[2]), (P(0.62f, 0.78f), Feet[3]) })
                {
                    p.BeginPath();
                    p.MoveTo(hip);
                    p.LineTo(P(foot.x, foot.y));
                    p.Stroke();
                }
            }
            for (var i = 0; i < 4; i++)
            {
                var c = FrictionZones.MuColour(mu[i]);
                c.a = 1f;
                var centre = P(Feet[i].x, Feet[i].y);
                p.fillColor = new Color(c.r, c.g, c.b, 0.25f);
                p.BeginPath();
                p.Arc(centre, 11f, Angle.Degrees(0f), Angle.Degrees(360f));
                p.Fill();
                p.fillColor = c;
                p.BeginPath();
                p.Arc(centre, 6f, Angle.Degrees(0f), Angle.Degrees(360f));
                p.Fill();
            }
        }
    }
}
