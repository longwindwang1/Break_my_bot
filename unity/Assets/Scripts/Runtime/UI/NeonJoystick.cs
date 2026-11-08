using System;
using UnityEngine;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// On-screen stick: drag the knob inside the ring; <see cref="Value"/> is in the unit disc (x right, y up) and
    /// springs back to zero on release. <see cref="Engaged"/> fires when a drag starts.
    /// </summary>
    public sealed class NeonJoystick : VisualElement
    {
        readonly VisualElement knob;
        bool dragging;

        public Vector2 Value { get; private set; }
        public event Action Engaged;

        public NeonJoystick()
        {
            AddToClassList("joystick");
            var cross = new VisualElement { pickingMode = PickingMode.Ignore };
            cross.AddToClassList("joystick-cross");
            Add(cross);
            knob = new VisualElement { pickingMode = PickingMode.Ignore };
            knob.AddToClassList("joystick-knob");
            Add(knob);

            RegisterCallback<PointerDownEvent>(e =>
            {
                dragging = true;
                this.CapturePointer(e.pointerId);
                Engaged?.Invoke();
                Move(e.localPosition);
                e.StopPropagation();
            });
            RegisterCallback<PointerMoveEvent>(e =>
            {
                if (dragging && this.HasPointerCapture(e.pointerId))
                    Move(e.localPosition);
            });
            RegisterCallback<PointerUpEvent>(e => Release(e.pointerId));
            RegisterCallback<PointerCaptureOutEvent>(_ => Release(-1));
        }

        void Move(Vector2 local)
        {
            var size = layout.size;
            var radius = Mathf.Min(size.x, size.y) / 2f;
            if (radius <= 0f)
                return;
            var d = (local - size / 2f) / radius;
            if (d.sqrMagnitude > 1f)
                d.Normalize();
            Value = new Vector2(d.x, -d.y);
            Place(d);
        }

        void Release(int pointerId)
        {
            if (pointerId >= 0 && this.HasPointerCapture(pointerId))
                this.ReleasePointer(pointerId);
            dragging = false;
            Value = Vector2.zero;
            Place(Vector2.zero);
        }

        void Place(Vector2 d)
        {
            // Knob centre in percent of the ring (the knob is centred with negative margins in USS).
            knob.style.left = Length.Percent(50f + d.x * 38f);
            knob.style.top = Length.Percent(50f + d.y * 38f);
        }
    }
}
