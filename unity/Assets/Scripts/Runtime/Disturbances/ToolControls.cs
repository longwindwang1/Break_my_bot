using System;
using UnityEngine;

namespace BMB
{
    /// <summary>
    /// A tool parameter or action, described for the UI: the game UI builds each tool's card from these, while the
    /// tool's keyboard / wheel shortcuts change the same properties, so both stay in sync. Values are in display
    /// units (e.g. cm, degrees); getters and setters convert.
    /// </summary>
    public abstract class ToolControl
    {
        public string Label = "";
        public string Tooltip = "";
        /// <summary>Shown only while this returns true (e.g. stairs-only parameters).</summary>
        public Func<bool> Visible = () => true;
    }

    /// <summary>A number with a range that depends on the limits mode, and the training range drawn on the track.</summary>
    public sealed class SliderControl : ToolControl
    {
        public Func<float> Get;
        public Action<float> Set;
        /// <summary>Allowed range in the active mode (training or hard).</summary>
        public Func<Vector2> Range;
        /// <summary>Range the policy was trained on (null: no marker, never "outside").</summary>
        public Func<Vector2> Training;
        public float Step = 0.01f;
        public string Format = "0.00";
        public string Unit = "";

        public bool OutsideTraining(float v)
        {
            if (Training == null)
                return false;
            var t = Training();
            return v < t.x - 1e-4f || v > t.y + 1e-4f;
        }

        /// <summary>Snaps to the step and clamps to the active range, then sets.</summary>
        public void SetClamped(float v)
        {
            var r = Range();
            if (Step > 0f)
                v = Mathf.Round(v / Step) * Step;
            Set(Mathf.Clamp(v, r.x, r.y));
        }

        public string Text(float v) => v.ToString(Format, System.Globalization.CultureInfo.InvariantCulture) + (Unit.Length > 0 ? " " + Unit : "");
    }

    /// <summary>One of several options (segmented buttons).</summary>
    public sealed class ChoiceControl : ToolControl
    {
        public string[] Options;
        public Func<int> Get;
        public Action<int> Set;
    }

    public sealed class ToggleControl : ToolControl
    {
        public Func<bool> Get;
        public Action<bool> Set;
    }

    public sealed class ButtonControl : ToolControl
    {
        public Action Click;
        /// <summary>Highlighted while true (e.g. "whole ground is ice").</summary>
        public Func<bool> Active = () => false;
        /// <summary>Destructive action (drawn in red).</summary>
        public bool Danger;
        /// <summary>Several buttons in a row share a group name.</summary>
        public string Row = "";
        /// <summary>Optional icon drawn before the label.</summary>
        public NeonIconKind? Icon;
    }

    /// <summary>A line of live text (e.g. what is on the robot now).</summary>
    public sealed class InfoControl : ToolControl
    {
        public Func<string> Text;
    }
}
