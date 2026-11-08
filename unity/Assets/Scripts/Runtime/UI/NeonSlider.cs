using UnityEngine;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// Slider for a <see cref="SliderControl"/>: label and value on top, a track with the training range shaded and its
    /// limit marked, a fill and a knob. Values beyond the training range turn the fill and the value red. Click or drag
    /// on the track; the active range (training or hard mode) and the value are re-read every <see cref="Refresh"/>.
    /// </summary>
    public sealed class NeonSlider : VisualElement
    {
        readonly SliderControl control;
        readonly Label value;
        readonly VisualElement track, training, fill, knob, marker;
        bool dragging;

        public SliderControl Control => control;

        public NeonSlider(SliderControl control)
        {
            this.control = control;
            AddToClassList("neon-slider");
            tooltip = control.Tooltip;

            var head = new VisualElement();
            head.AddToClassList("slider-head");
            head.Add(new Label(control.Label) { name = "label", pickingMode = PickingMode.Ignore });
            value = new Label { name = "value", pickingMode = PickingMode.Ignore };
            value.AddToClassList("num");
            head.Add(value);
            Add(head);

            track = new VisualElement { name = "track" };
            track.AddToClassList("slider-track");
            Child(track, "slider-rail");
            training = Child(track, "slider-training");
            fill = Child(track, "slider-fill");
            marker = Child(track, "slider-marker");
            knob = Child(track, "slider-knob");
            Add(track);

            track.RegisterCallback<PointerDownEvent>(e =>
            {
                dragging = true;
                track.CapturePointer(e.pointerId);
                SetFromLocalX(e.localPosition.x);
                e.StopPropagation();
            });
            track.RegisterCallback<PointerMoveEvent>(e =>
            {
                if (dragging && track.HasPointerCapture(e.pointerId))
                    SetFromLocalX(e.localPosition.x);
            });
            track.RegisterCallback<PointerUpEvent>(e =>
            {
                dragging = false;
                track.ReleasePointer(e.pointerId);
            });
            Refresh();
        }

        static VisualElement Child(VisualElement parent, string cls)
        {
            var c = new VisualElement { pickingMode = PickingMode.Ignore };
            c.AddToClassList(cls);
            parent.Add(c);
            return c;
        }

        void SetFromLocalX(float x)
        {
            var w = track.resolvedStyle.width;
            if (w <= 0f || float.IsNaN(w))
                return;
            var r = control.Range();
            control.SetClamped(Mathf.Lerp(r.x, r.y, Mathf.Clamp01(x / w)));
            Refresh();
        }

        /// <summary>Position of a value along the active range, 0..1.</summary>
        public static float Fraction(float v, Vector2 range) => range.y > range.x ? Mathf.Clamp01((v - range.x) / (range.y - range.x)) : 0f;

        public void Refresh()
        {
            var r = control.Range();
            var v = Mathf.Clamp(control.Get(), r.x, r.y);
            var f = Fraction(v, r);
            fill.style.width = Length.Percent(f * 100f);
            knob.style.left = Length.Percent(f * 100f);
            value.text = control.Text(control.Get());

            if (control.Training != null)
            {
                var t = control.Training();
                var lo = Fraction(t.x, r);
                var hi = Fraction(t.y, r);
                training.style.display = DisplayStyle.Flex;
                training.style.left = Length.Percent(lo * 100f);
                training.style.width = Length.Percent((hi - lo) * 100f);
                var markerVisible = t.y < r.y - 1e-4f;
                marker.style.display = markerVisible ? DisplayStyle.Flex : DisplayStyle.None;
                marker.style.left = Length.Percent(hi * 100f);
            }
            else
            {
                training.style.display = DisplayStyle.None;
                marker.style.display = DisplayStyle.None;
            }
            EnableInClassList("outside", control.OutsideTraining(v));
        }
    }
}
