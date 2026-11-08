using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace BMB
{
    public enum NeonIconKind
    {
        Push, Drag, Ball, Payload, Friction, Obstacle, Deceptive,
        Reset, Pause, Play, Help, Hard, Export, TurnLeft, TurnRight, Lock, Close,
    }

    /// <summary>
    /// Vector icon drawn with Painter2D: three strokes (wide and faint → thin and bright) give a neon glow without
    /// textures. The colour comes from the USS custom property --icon-color, so hover / selected states restyle it.
    /// Shapes are in a unit square (0..1, y down).
    /// </summary>
    public sealed class NeonIcon : VisualElement
    {
        static readonly CustomStyleProperty<Color> IconColorProperty = new CustomStyleProperty<Color>("--icon-color");

        NeonIconKind kind;
        Color colour = new Color(0f, 0.9f, 1f);

        public NeonIconKind Kind
        {
            get => kind;
            set
            {
                kind = value;
                MarkDirtyRepaint();
            }
        }

        public NeonIcon() : this(NeonIconKind.Lock) { }

        public NeonIcon(NeonIconKind kind)
        {
            this.kind = kind;
            AddToClassList("neon-icon");
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
            RegisterCallback<CustomStyleResolvedEvent>(e =>
            {
                if (e.customStyle.TryGetValue(IconColorProperty, out var c) && c != colour)
                {
                    colour = c;
                    MarkDirtyRepaint();
                }
            });
        }

        void Draw(MeshGenerationContext ctx)
        {
            var r = contentRect;
            var s = Mathf.Min(r.width, r.height);
            if (s < 2f)
                return;
            var p = ctx.painter2D;
            var origin = new Vector2(r.x + (r.width - s) / 2f, r.y + (r.height - s) / 2f);
            Vector2 P(float x, float y) => origin + new Vector2(x, y) * s;

            // Glow passes: (width as a fraction of the size, alpha).
            foreach (var (w, a) in new[] { (0.16f, 0.12f), (0.09f, 0.3f), (0.045f, 1f) })
            {
                p.strokeColor = new Color(colour.r, colour.g, colour.b, colour.a * a);
                p.lineWidth = Mathf.Max(1f, w * s);
                p.lineCap = LineCap.Round;
                p.lineJoin = LineJoin.Round;
                Shape(p, P, s);
            }
        }

        void Shape(Painter2D p, Func<float, float, Vector2> P, float s)
        {
            void Line(params Vector2[] pts)
            {
                p.BeginPath();
                p.MoveTo(pts[0]);
                for (var i = 1; i < pts.Length; i++)
                    p.LineTo(pts[i]);
                p.Stroke();
            }
            void Closed(params Vector2[] pts)
            {
                p.BeginPath();
                p.MoveTo(pts[0]);
                for (var i = 1; i < pts.Length; i++)
                    p.LineTo(pts[i]);
                p.ClosePath();
                p.Stroke();
            }
            void Circle(float x, float y, float radius)
            {
                p.BeginPath();
                p.Arc(P(x, y), radius * s, Angle.Degrees(0f), Angle.Degrees(360f));
                p.ClosePath();
                p.Stroke();
            }
            void ArcStroke(float x, float y, float radius, float from, float to, ArcDirection dir = ArcDirection.Clockwise)
            {
                p.BeginPath();
                p.Arc(P(x, y), radius * s, Angle.Degrees(from), Angle.Degrees(to), dir);
                p.Stroke();
            }

            switch (kind)
            {
                case NeonIconKind.Push:
                    Line(P(0.3f, 0.5f), P(0.85f, 0.5f));
                    Line(P(0.65f, 0.3f), P(0.85f, 0.5f), P(0.65f, 0.7f));
                    Line(P(0.1f, 0.32f), P(0.3f, 0.32f));
                    Line(P(0.15f, 0.68f), P(0.3f, 0.68f));
                    break;
                case NeonIconKind.Drag:
                    Line(P(0.15f, 0.15f), P(0.5f, 0.5f));
                    Circle(0.62f, 0.62f, 0.18f);
                    Circle(0.62f, 0.62f, 0.03f);
                    break;
                case NeonIconKind.Ball:
                    Circle(0.62f, 0.52f, 0.22f);
                    Line(P(0.08f, 0.38f), P(0.3f, 0.38f));
                    Line(P(0.12f, 0.55f), P(0.32f, 0.55f));
                    Line(P(0.08f, 0.72f), P(0.28f, 0.72f));
                    break;
                case NeonIconKind.Payload:
                    Closed(P(0.2f, 0.85f), P(0.8f, 0.85f), P(0.68f, 0.4f), P(0.32f, 0.4f));
                    ArcStroke(0.5f, 0.32f, 0.1f, 180f, 360f);
                    Line(P(0.42f, 0.62f), P(0.58f, 0.62f));
                    break;
                case NeonIconKind.Friction:
                    for (var k = 0; k < 3; k++)
                    {
                        var ang = k * Mathf.PI / 3f + Mathf.PI / 2f;
                        var d = new Vector2(Mathf.Cos(ang), Mathf.Sin(ang)) * 0.38f;
                        Line(P(0.5f - d.x, 0.5f - d.y), P(0.5f + d.x, 0.5f + d.y));
                        foreach (var sgn in new[] { -1f, 1f })
                        {
                            var tip = new Vector2(0.5f, 0.5f) + d * sgn * 0.75f;
                            var n = new Vector2(-d.y, d.x) * 0.3f;
                            Line(P(tip.x + n.x + d.x * sgn * 0.3f, tip.y + n.y + d.y * sgn * 0.3f), P(tip.x, tip.y),
                                 P(tip.x - n.x + d.x * sgn * 0.3f, tip.y - n.y + d.y * sgn * 0.3f));
                        }
                    }
                    break;
                case NeonIconKind.Obstacle:
                    Closed(P(0.12f, 0.85f), P(0.12f, 0.65f), P(0.38f, 0.65f), P(0.38f, 0.45f), P(0.62f, 0.45f), P(0.62f, 0.25f),
                           P(0.88f, 0.25f), P(0.88f, 0.85f));
                    break;
                case NeonIconKind.Deceptive:
                    Closed(P(0.5f, 0.13f), P(0.9f, 0.83f), P(0.1f, 0.83f));
                    Line(P(0.5f, 0.38f), P(0.5f, 0.6f));
                    Circle(0.5f, 0.71f, 0.02f);
                    break;
                case NeonIconKind.Reset:
                    ArcStroke(0.5f, 0.52f, 0.3f, -60f, 220f);
                    Line(P(0.62f, 0.12f), P(0.66f, 0.27f), P(0.5f, 0.3f));
                    break;
                case NeonIconKind.Pause:
                    Line(P(0.36f, 0.22f), P(0.36f, 0.78f));
                    Line(P(0.64f, 0.22f), P(0.64f, 0.78f));
                    break;
                case NeonIconKind.Play:
                    Closed(P(0.32f, 0.2f), P(0.8f, 0.5f), P(0.32f, 0.8f));
                    break;
                case NeonIconKind.Help:
                    Circle(0.5f, 0.5f, 0.38f);
                    ArcStroke(0.5f, 0.4f, 0.12f, 180f, 405f);
                    Line(P(0.585f, 0.485f), P(0.5f, 0.56f), P(0.5f, 0.6f));
                    Circle(0.5f, 0.7f, 0.015f);
                    break;
                case NeonIconKind.Hard:
                    Closed(P(0.58f, 0.08f), P(0.26f, 0.56f), P(0.48f, 0.56f), P(0.4f, 0.92f), P(0.74f, 0.42f), P(0.52f, 0.42f));
                    break;
                case NeonIconKind.Export:
                    Line(P(0.5f, 0.15f), P(0.5f, 0.62f));
                    Line(P(0.32f, 0.45f), P(0.5f, 0.63f), P(0.68f, 0.45f));
                    Line(P(0.18f, 0.66f), P(0.18f, 0.85f), P(0.82f, 0.85f), P(0.82f, 0.66f));
                    break;
                case NeonIconKind.TurnLeft:
                    ArcStroke(0.5f, 0.58f, 0.28f, 0f, -180f, ArcDirection.CounterClockwise);
                    Line(P(0.08f, 0.45f), P(0.22f, 0.6f), P(0.36f, 0.45f));
                    break;
                case NeonIconKind.TurnRight:
                    ArcStroke(0.5f, 0.58f, 0.28f, 180f, 360f);
                    Line(P(0.64f, 0.45f), P(0.78f, 0.6f), P(0.92f, 0.45f));
                    break;
                case NeonIconKind.Lock:
                    Closed(P(0.25f, 0.48f), P(0.75f, 0.48f), P(0.75f, 0.85f), P(0.25f, 0.85f));
                    ArcStroke(0.5f, 0.48f, 0.17f, 180f, 360f);
                    break;
                case NeonIconKind.Close:
                    Line(P(0.25f, 0.25f), P(0.75f, 0.75f));
                    Line(P(0.75f, 0.25f), P(0.25f, 0.75f));
                    break;
            }
        }
    }
}
