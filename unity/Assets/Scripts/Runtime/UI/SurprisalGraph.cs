using UnityEngine;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// Scrolling plot of the world model's surprise over the last 5 s (one sample per policy step): the standardized
    /// score x = (S − m0) / s0 as a cyan line, the CUSUM g / h as a magenta area (an alarm when it reaches the top),
    /// and a tick at every alarm.
    /// </summary>
    public sealed class SurprisalGraph : VisualElement
    {
        public const int Capacity = 250;  // 5 s at 50 Hz
        const float XMin = -2f, XMax = 10f;

        static readonly Color Cyan = new Color(0f, 0.9f, 1f, 0.95f);
        static readonly Color Area = new Color(1f, 0.17f, 0.84f, 0.22f);
        static readonly Color AreaEdge = new Color(1f, 0.17f, 0.84f, 0.7f);
        static readonly Color Alarm = new Color(1f, 0.17f, 0.84f, 0.95f);
        static readonly Color Grid = new Color(0.43f, 0.55f, 0.63f, 0.35f);

        readonly float[] x = new float[Capacity], cusum = new float[Capacity];
        readonly bool[] alarm = new bool[Capacity];
        int head, count;

        public int Count => count;

        public SurprisalGraph()
        {
            AddToClassList("wm-graph");
            pickingMode = PickingMode.Ignore;
            generateVisualContent += Draw;
        }

        /// <param name="score">Standardized surprise x (σ units; 0 = typical of clean walking).</param>
        /// <param name="cusumRatio">CUSUM g / h before an alarm reset (1 = alarm).</param>
        public void Push(float score, float cusumRatio, bool isAlarm)
        {
            x[head] = score;
            cusum[head] = cusumRatio;
            alarm[head] = isAlarm;
            head = (head + 1) % Capacity;
            count = Mathf.Min(count + 1, Capacity);
            MarkDirtyRepaint();
        }

        public void ClearSamples()
        {
            head = count = 0;
            MarkDirtyRepaint();
        }

        int Index(int i) => (head - count + i + Capacity) % Capacity;  // i = 0 oldest

        void Draw(MeshGenerationContext ctx)
        {
            var r = contentRect;
            if (r.width < 4f || r.height < 4f)
                return;
            var p = ctx.painter2D;
            float X(int i) => r.width * (1f - (count - 1 - i) / (float)(Capacity - 1));
            float Y(float v) => r.height * (1f - (Mathf.Clamp(v, XMin, XMax) - XMin) / (XMax - XMin));

            p.lineWidth = 1f;
            p.strokeColor = Grid;
            p.BeginPath();
            p.MoveTo(new Vector2(0f, Y(0f)));
            p.LineTo(new Vector2(r.width, Y(0f)));
            p.Stroke();
            if (count < 2)
                return;

            // CUSUM area, bottom-anchored, full height = threshold
            p.fillColor = Area;
            p.BeginPath();
            p.MoveTo(new Vector2(X(0), r.height));
            for (var i = 0; i < count; i++)
                p.LineTo(new Vector2(X(i), r.height * (1f - Mathf.Clamp01(cusum[Index(i)]))));
            p.LineTo(new Vector2(X(count - 1), r.height));
            p.ClosePath();
            p.Fill();
            p.strokeColor = AreaEdge;
            p.BeginPath();
            for (var i = 0; i < count; i++)
            {
                var pt = new Vector2(X(i), r.height * (1f - Mathf.Clamp01(cusum[Index(i)])));
                if (i == 0) p.MoveTo(pt); else p.LineTo(pt);
            }
            p.Stroke();

            // alarms
            p.strokeColor = Alarm;
            p.lineWidth = 2f;
            for (var i = 0; i < count; i++)
                if (alarm[Index(i)])
                {
                    p.BeginPath();
                    p.MoveTo(new Vector2(X(i), 0f));
                    p.LineTo(new Vector2(X(i), r.height));
                    p.Stroke();
                }

            // surprise line
            p.strokeColor = Cyan;
            p.lineWidth = 1.5f;
            p.lineJoin = LineJoin.Round;
            p.BeginPath();
            for (var i = 0; i < count; i++)
            {
                var pt = new Vector2(X(i), Y(x[Index(i)]));
                if (i == 0) p.MoveTo(pt); else p.LineTo(pt);
            }
            p.Stroke();
        }
    }
}
