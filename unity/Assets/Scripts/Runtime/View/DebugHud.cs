using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.InputSystem;

namespace BMB
{
    /// <summary>
    /// Plain on-screen text: limits mode, active tool and its parameters, command, base state, subsystem lines
    /// (payload, foot friction, …), falls and the latest events. Red = outside the v2 training ranges. F1 toggles help.
    /// </summary>
    public sealed class DebugHud : MonoBehaviour
    {
        [SerializeField] DisturbanceManager manager;
        [SerializeField] CommandSource commandSource;
        [SerializeField] int recentEvents = 6;
        [SerializeField] int fontSize = 14;

        bool showHelp = true;
        GUIStyle style;
        readonly StringBuilder sb = new StringBuilder();

        const string Red = "#ff5a4a", Dim = "#b8c0c8", Green = "#7be08a";

        public void Configure(DisturbanceManager disturbanceManager, CommandSource commands)
        {
            manager = disturbanceManager;
            commandSource = commands;
        }

        void Update()
        {
            if (Keyboard.current != null && Keyboard.current.f1Key.wasPressedThisFrame)
                showHelp = !showHelp;
        }

        void OnGUI()
        {
            if (manager == null || manager.Controller == null || !manager.Controller.Initialized)
                return;
            style ??= new GUIStyle(GUI.skin.label) { richText = true, fontSize = fontSize, wordWrap = false };
            var c = manager.Controller;

            sb.Clear();
            sb.AppendLine(manager.HardMode
                ? $"<b><color={Red}>HARD MODE</color></b> <color={Dim}>(beyond training ranges; H to switch back)</color>"
                : $"<b><color={Green}>Training ranges</color></b> <color={Dim}>(H: hard mode)</color>");

            var tool = manager.ActiveTool;
            sb.AppendLine(tool == null ? $"Tool: none <color={Dim}>(keys 1–7)</color>" : $"Tool: <b>{tool.Name}</b>  {tool.Status(manager)}");
            if (tool != null && tool.Help.Length > 0)
                sb.AppendLine($"<color={Dim}>  {tool.Help}</color>");

            var vBase = IsaacFrame.PolarInBase(c.Body.BaseRotation, c.Body.BaseComLinearVelocity);
            var fixedCmd = commandSource != null && commandSource.UseFixedCommand;
            sb.AppendLine($"Policy {c.Variant}   command ({c.Command.x:F2}, {c.Command.y:F2}, {c.Command.z:F2}) {(fixedCmd ? "fixed" : "manual")}");
            sb.AppendLine($"Base: v ({vBase.x:F2}, {vBase.y:F2}) m/s   height {c.BaseHeight:F2} m   t {c.SimTime:F1} s   falls {c.Falls}");
            foreach (var line in manager.StatusLines)
                sb.AppendLine(line());

            var events = manager.Log.Events;
            for (var i = Mathf.Max(0, events.Count - recentEvents); i < events.Count; i++)
            {
                var e = events[i];
                var colour = e.OutsideTraining || e.Type == "fall" ? Red : Dim;
                sb.AppendLine($"<color={colour}>{e.Time,7:F1} s  {e.Type}: {e.Detail}</color>");
            }

            if (showHelp)
            {
                sb.AppendLine();
                sb.AppendLine($"<color={Dim}>Robot: W/S A/D Q/E move · Tab fixed/manual · R reset   Camera: hold right mouse to orbit</color>");
                var keys = new List<string>();
                for (var i = 0; i < manager.Tools.Count; i++)
                    keys.Add($"{i + 1} {manager.Tools[i].Name}");
                sb.AppendLine($"<color={Dim}>Tools: {string.Join(" · ", keys)} · Esc none · H hard mode · F1 hide help</color>");
            }

            var text = sb.ToString();
            var size = style.CalcSize(new GUIContent(text));
            var rect = new Rect(10, 10, size.x + 16, size.y + 12);
            GUI.color = new Color(0f, 0f, 0f, 0.55f);
            GUI.DrawTexture(rect, Texture2D.whiteTexture);
            GUI.color = Color.white;
            GUI.Label(new Rect(rect.x + 8, rect.y + 6, size.x, size.y), text, style);
        }
    }
}
