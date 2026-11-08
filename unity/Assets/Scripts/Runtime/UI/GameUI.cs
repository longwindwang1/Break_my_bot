using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UIElements;

namespace BMB
{
    /// <summary>
    /// The game HUD (UI Toolkit, GameHud.uxml + Neon.uss): top bar, tool rail with a card built from each tool's
    /// <see cref="ToolControl"/>s, robot status and event feed, drive mode, on-screen stick and turn buttons, help.
    /// Everything the keyboard shortcuts do is reachable here with the mouse. While the mouse is over a panel the
    /// disturbance tools ignore it (<see cref="DisturbanceManager.PointerOverUI"/>).
    /// </summary>
    [RequireComponent(typeof(UIDocument))]
    public sealed class GameUI : MonoBehaviour
    {
        [SerializeField] DisturbanceManager manager;
        [SerializeField] CommandSource commands;
        [Tooltip("Show the controls overlay when play starts")]
        [SerializeField] bool showHelpOnStart = true;

        VisualElement root, toolCard, toolControls, toolIconSlot, eventsList, help, driveDetail;
        Label title, policy, clock, falls, toolTitle, toolHint, speedValue, heightValue, tiltValue, payloadValue, tooltip, toast;
        Button modeButton, pauseButton, turnLeft, turnRight;
        VisualElement speedCmd, speedActual;
        FootDiagram feet;
        NeonJoystick stick;
        readonly List<VisualElement> railButtons = new List<VisualElement>();
        readonly Dictionary<DisturbanceTool, VisualElement> cards = new Dictionary<DisturbanceTool, VisualElement>();
        readonly List<(ToolControl control, VisualElement element)> bound = new List<(ToolControl, VisualElement)>();
        readonly List<(Button button, ToolControl control, Func<bool> on)> segments = new List<(Button, ToolControl, Func<bool>)>();
        Button fixedButton, manualButton;
        NeonSlider cruise;
        DisturbanceTool shownTool;
        int shownEvents = -1;
        float toastUntil;
        float turnHeld;

        public VisualElement Root => root;
        public bool HelpVisible => help != null && help.style.display != DisplayStyle.None;

        void OnEnable()
        {
            var doc = GetComponent<UIDocument>();
            if (doc != null && doc.rootVisualElement != null && manager != null)
                Bind(doc.rootVisualElement, manager, commands);
        }

        void OnDisable()
        {
            if (manager != null)
                manager.PointerOverUI = false;
        }

        /// <summary>Wires a cloned GameHud tree to the game (also used by tests without a panel).</summary>
        public void Bind(VisualElement treeRoot, DisturbanceManager disturbanceManager, CommandSource commandSource)
        {
            manager = disturbanceManager;
            commands = commandSource;
            root = treeRoot.Q("root") ?? treeRoot;
            cards.Clear();
            railButtons.Clear();
            bound.Clear();
            segments.Clear();
            shownTool = null;
            shownEvents = -1;

            title = root.Q<Label>("title");
            policy = root.Q<Label>("policy");
            clock = root.Q<Label>("clock");
            falls = root.Q<Label>("falls");
            modeButton = root.Q<Button>("mode");
            modeButton.clicked += () => manager.HardMode = !manager.HardMode;
            Tip(modeButton, "Training ranges vs hard mode (H)");

            BuildTopButtons(root.Q("top-buttons"));
            BuildRail(root.Q("rail"));

            toolCard = root.Q("tool-card");
            toolControls = root.Q("tool-controls");
            toolIconSlot = root.Q("tool-icon-slot");
            toolTitle = root.Q<Label>("tool-title");
            toolHint = root.Q<Label>("tool-hint");

            speedValue = root.Q<Label>("speed-value");
            heightValue = root.Q<Label>("height-value");
            tiltValue = root.Q<Label>("tilt-value");
            payloadValue = root.Q<Label>("payload-value");
            speedCmd = root.Q("speed-cmd");
            speedActual = root.Q("speed-actual");
            feet = new FootDiagram();
            root.Q("feet-slot").Add(feet);
            eventsList = root.Q("events");

            BuildDrive(root.Q("drive-mode"), root.Q("drive-detail"));
            BuildStick();
            BuildHelp(root.Q("help-columns"));
            help = root.Q("help");
            root.Q<Button>("help-close").Add(new NeonIcon(NeonIconKind.Close));
            root.Q<Button>("help-close").clicked += () => ShowHelp(false);
            help.RegisterCallback<PointerDownEvent>(e =>
            {
                if (e.target == help)
                    ShowHelp(false);
            });
            ShowHelp(showHelpOnStart && Application.isPlaying);

            tooltip = root.Q<Label>("tooltip");
            toast = root.Q<Label>("toast");
            UiFonts.Apply(root);
            Refresh();
        }

        // --- building ------------------------------------------------------------------------------------------------

        Button IconButton(NeonIconKind icon, string tip, Action click, string cls = "icon-btn")
        {
            var b = new Button(click);
            b.AddToClassList(cls);
            b.Add(new NeonIcon(icon));
            Tip(b, tip);
            return b;
        }

        void BuildTopButtons(VisualElement bar)
        {
            bar.Clear();
            bar.Add(IconButton(NeonIconKind.Reset, "Reset the robot (R)", () => manager.Controller.ResetRobot()));
            pauseButton = IconButton(NeonIconKind.Pause, "Pause / resume the simulation", TogglePause);
            bar.Add(pauseButton);
            bar.Add(IconButton(NeonIconKind.Export, "Save the event log as CSV", ExportLog));
            bar.Add(IconButton(NeonIconKind.Help, "Controls (F1)", () => ShowHelp(!HelpVisible)));
        }

        void BuildRail(VisualElement rail)
        {
            rail.Clear();
            for (var i = 0; i < manager.Tools.Count; i++)
            {
                var tool = manager.Tools[i];
                var b = new VisualElement();
                b.AddToClassList("rail-btn");
                b.Add(new NeonIcon(tool.Icon));
                var key = new Label((i + 1).ToString()) { pickingMode = PickingMode.Ignore };
                key.AddToClassList("rail-key");
                b.Add(key);
                if (!tool.Available)
                {
                    b.AddToClassList("locked");
                    var soon = new Label("SOON") { pickingMode = PickingMode.Ignore };
                    soon.AddToClassList("rail-soon");
                    b.Add(soon);
                }
                Tip(b, tool.Available ? $"{tool.Name}  ({i + 1})" : $"{tool.Name} — coming soon");
                b.RegisterCallback<ClickEvent>(_ =>
                {
                    if (!tool.Available)
                        ShowToast($"{tool.Name}: coming soon");
                    else
                        manager.Select(manager.ActiveTool == tool ? null : tool);
                });
                railButtons.Add(b);
                rail.Add(b);
            }
        }

        VisualElement BuildCard(DisturbanceTool tool)
        {
            var card = new VisualElement();
            VisualElement row = null;
            var rowName = "";
            foreach (var control in tool.BuildControls(manager))
            {
                VisualElement element;
                switch (control)
                {
                    case SliderControl s:
                        element = new NeonSlider(s);
                        break;
                    case ChoiceControl c:
                    {
                        element = new VisualElement();
                        element.Add(FieldLabel(c.Label));
                        var segs = new VisualElement();
                        segs.AddToClassList("seg-row");
                        for (var i = 0; i < c.Options.Length; i++)
                        {
                            var index = i;
                            var b = new Button(() => c.Set(index)) { text = c.Options[i] };
                            b.AddToClassList("seg");
                            segments.Add((b, c, () => c.Get() == index));
                            segs.Add(b);
                        }
                        element.Add(segs);
                        break;
                    }
                    case ToggleControl t:
                    {
                        var b = new Button(() => t.Set(!t.Get())) { text = t.Label };
                        b.AddToClassList("seg");
                        segments.Add((b, t, () => t.Get()));
                        element = b;
                        break;
                    }
                    case ButtonControl bc:
                    {
                        var b = new Button(bc.Click);
                        b.AddToClassList("btn");
                        if (bc.Danger)
                            b.AddToClassList("danger");
                        if (bc.Icon.HasValue)
                            b.Add(new NeonIcon(bc.Icon.Value));
                        b.Add(new Label(bc.Label) { pickingMode = PickingMode.Ignore });
                        segments.Add((b, bc, bc.Active));
                        element = b;
                        break;
                    }
                    case InfoControl info:
                        element = new Label { pickingMode = PickingMode.Ignore };
                        element.AddToClassList("info");
                        break;
                    default:
                        continue;
                }
                if (control.Tooltip.Length > 0)
                    Tip(element, control.Tooltip);
                bound.Add((control, element));

                // Buttons sharing a Row name sit side by side.
                var groupName = control is ButtonControl btn ? btn.Row : "";
                if (groupName.Length > 0)
                {
                    if (row == null || rowName != groupName)
                    {
                        row = new VisualElement();
                        row.AddToClassList("btn-row");
                        card.Add(row);
                        rowName = groupName;
                    }
                    row.Add(element);
                }
                else
                {
                    row = null;
                    rowName = "";
                    card.Add(element);
                }
            }
            UiFonts.ApplyMono(card);
            return card;
        }

        static Label FieldLabel(string text)
        {
            var l = new Label(text) { pickingMode = PickingMode.Ignore };
            l.AddToClassList("field-label");
            return l;
        }

        void BuildDrive(VisualElement mode, VisualElement detail)
        {
            driveDetail = detail;
            mode.Clear();
            fixedButton = new Button(() => commands.SetFixed(true)) { text = "CRUISE" };
            manualButton = new Button(() => commands.SetFixed(false)) { text = "MANUAL" };
            foreach (var b in new[] { fixedButton, manualButton })
            {
                b.AddToClassList("seg");
                mode.Add(b);
            }
            Tip(fixedButton, "Walk straight at a set speed (Tab)");
            Tip(manualButton, "Steer with the stick or W/S A/D Q/E (Tab)");
            cruise = new NeonSlider(new SliderControl
            {
                Label = "Cruise speed", Unit = "m/s", Format = "0.0", Step = 0.1f,
                Get = () => commands.FixedSpeed, Set = v => commands.FixedSpeed = v,
                Range = () => new Vector2(commands.Ranges.LinVelX[0], commands.Ranges.LinVelX[1]),
            });
            detail.Add(cruise);
            var keys = new Label("Stick or W/S A/D · Q/E turn") { name = "manual-hint", pickingMode = PickingMode.Ignore };
            keys.AddToClassList("info");
            detail.Add(keys);
        }

        void BuildStick()
        {
            stick = new NeonJoystick();
            root.Q("stick-slot").Add(stick);
            stick.Engaged += () => commands.SetFixed(false);
            turnLeft = root.Q<Button>("turn-left");
            turnRight = root.Q<Button>("turn-right");
            turnLeft.Add(new NeonIcon(NeonIconKind.TurnLeft));
            turnRight.Add(new NeonIcon(NeonIconKind.TurnRight));
            Tip(turnLeft, "Turn left (Q)");
            Tip(turnRight, "Turn right (E)");
            foreach (var (button, sign) in new[] { (turnLeft, 1f), (turnRight, -1f) })
            {
                button.RegisterCallback<PointerDownEvent>(_ =>
                {
                    commands.SetFixed(false);
                    turnHeld = sign;
                    button.AddToClassList("held");
                }, TrickleDown.TrickleDown);
                button.RegisterCallback<PointerUpEvent>(_ => Release(button));
                button.RegisterCallback<PointerLeaveEvent>(_ => Release(button));
            }
            void Release(Button b)
            {
                turnHeld = 0f;
                b.RemoveFromClassList("held");
            }
        }

        void BuildHelp(VisualElement columns)
        {
            columns.Clear();
            var sections = new[]
            {
                ("MOUSE", new[]
                {
                    ("Left click", "Use the selected tool"), ("Right drag", "Orbit the camera"), ("Ctrl + wheel", "Zoom"),
                    ("Wheel", "Main parameter of the tool"), ("Stick / arrows", "Drive the robot (bottom right)"),
                }),
                ("KEYS", new[]
                {
                    ("1 – 7", "Select a tool · Esc none"), ("W/S A/D Q/E", "Drive · Tab cruise/manual"), ("R", "Reset the robot"),
                    ("H", "Training ranges / hard mode"), ("F1", "This help"),
                }),
                ("TOOLS", new[]
                {
                    ("Z / Shift+Z", "Obstacle kind"), ("[  ]", "Brush or footprint size"), (",  .", "Rotate obstacle"),
                    ("X / Shift+X", "Remove obstacle / all"), ("G · Backspace", "Whole ground μ · clear"),
                }),
            };
            foreach (var (name, rows) in sections)
            {
                var col = new VisualElement();
                col.AddToClassList("help-column");
                var head = new Label(name) { pickingMode = PickingMode.Ignore };
                head.AddToClassList("help-section");
                col.Add(head);
                foreach (var (key, text) in rows)
                {
                    var r = new VisualElement { pickingMode = PickingMode.Ignore };
                    r.AddToClassList("help-row");
                    var k = new Label(key) { pickingMode = PickingMode.Ignore };
                    k.AddToClassList("help-key");
                    var t = new Label(text) { pickingMode = PickingMode.Ignore };
                    t.AddToClassList("help-text");
                    r.Add(k);
                    r.Add(t);
                    col.Add(r);
                }
                columns.Add(col);
            }
        }

        // --- behaviour -----------------------------------------------------------------------------------------------

        void TogglePause()
        {
            Time.timeScale = Time.timeScale > 0f ? 0f : 1f;
            ShowToast(Time.timeScale > 0f ? "Resumed" : "Paused");
        }

        void ExportLog()
        {
            var path = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Logs", $"disturbances_{DateTime.Now:yyyyMMdd_HHmmss}.csv"));
            manager.Log.WriteCsv(path);
            ShowToast($"Saved {manager.Log.Events.Count} events to Logs/{Path.GetFileName(path)}");
        }

        public void ShowHelp(bool show)
        {
            if (help != null)
                help.style.display = show ? DisplayStyle.Flex : DisplayStyle.None;
        }

        void ShowToast(string text)
        {
            if (toast == null)
                return;
            toast.text = text;
            toast.AddToClassList("show");
            toastUntil = Time.unscaledTime + 2.5f;
        }

        void Tip(VisualElement e, string text)
        {
            e.RegisterCallback<PointerEnterEvent>(ev =>
            {
                if (tooltip == null)
                    return;
                tooltip.text = text;
                tooltip.style.display = DisplayStyle.Flex;
                var p = ev.position;
                tooltip.style.left = p.x + 16f;
                tooltip.style.top = p.y + 18f;
            });
            e.RegisterCallback<PointerLeaveEvent>(_ =>
            {
                if (tooltip != null)
                    tooltip.style.display = DisplayStyle.None;
            });
        }

        void Update()
        {
            if (root == null || manager == null)
                return;
            UpdatePointer();
            if (Keyboard.current != null && Keyboard.current.f1Key.wasPressedThisFrame)
                ShowHelp(!HelpVisible);
            if (commands != null)
            {
                commands.UiStick = stick != null ? stick.Value : Vector2.zero;
                commands.UiTurn = turnHeld;
            }
            if (toast != null && toastUntil > 0f && Time.unscaledTime > toastUntil)
            {
                toast.RemoveFromClassList("show");
                toastUntil = 0f;
            }
            Refresh();
        }

        void UpdatePointer()
        {
            var mouse = Mouse.current;
            if (mouse == null || root.panel == null)
                return;
            var screen = mouse.position.ReadValue();
            var panelPos = RuntimePanelUtils.ScreenToPanel(root.panel, new Vector2(screen.x, Screen.height - screen.y));
            var picked = root.panel.Pick(panelPos);
            manager.PointerOverUI = picked != null || HelpVisible;
        }

        /// <summary>Re-reads every value shown (called each frame).</summary>
        public void Refresh()
        {
            var c = manager.Controller;
            var ready = c != null && c.Initialized;

            modeButton.text = manager.HardMode ? "HARD MODE" : "TRAINING RANGE";
            modeButton.EnableInClassList("hard", manager.HardMode);
            policy.text = c != null ? $"POLICY {c.Variant.ToUpperInvariant()}" : "POLICY —";
            if (ready)
            {
                clock.text = $"T {c.SimTime:0.0} s";
                falls.text = $"FALLS {c.Falls}";
                falls.EnableInClassList("alert", c.Falls > 0);
            }
            if (pauseButton != null)
                pauseButton.Q<NeonIcon>().Kind = Time.timeScale > 0f ? NeonIconKind.Pause : NeonIconKind.Play;

            RefreshTool();
            if (ready)
                RefreshRobot(c);
            RefreshEvents();
            RefreshDrive();
        }

        void RefreshTool()
        {
            var tool = manager.ActiveTool;
            for (var i = 0; i < railButtons.Count && i < manager.Tools.Count; i++)
                railButtons[i].EnableInClassList("on", manager.Tools[i] == tool);

            if (tool != shownTool)
            {
                shownTool = tool;
                toolControls.Clear();
                toolIconSlot.Clear();
                if (tool != null)
                {
                    if (!cards.TryGetValue(tool, out var card))
                        cards[tool] = card = BuildCard(tool);
                    toolControls.Add(card);
                    toolIconSlot.Add(new NeonIcon(tool.Icon));
                    toolTitle.text = tool.Name.ToUpperInvariant();
                }
            }
            toolCard.style.display = tool != null ? DisplayStyle.Flex : DisplayStyle.None;
            if (tool == null)
                return;
            toolHint.text = tool.Hint;

            foreach (var (control, element) in bound)
            {
                if (!IsInCard(element))
                    continue;
                element.style.display = control.Visible() ? DisplayStyle.Flex : DisplayStyle.None;
                switch (element)
                {
                    case NeonSlider s:
                        s.Refresh();
                        break;
                    case Label l when control is InfoControl info:
                        l.text = info.Text();
                        break;
                }
            }
            foreach (var (button, control, on) in segments)
                if (IsInCard(button))
                    button.EnableInClassList(control is ButtonControl ? "active" : "on", on());
        }

        bool IsInCard(VisualElement e) => e.panel == toolControls.panel && toolControls.Contains(e);

        void RefreshRobot(Go2Controller c)
        {
            var v = IsaacFrame.PolarInBase(c.Body.BaseRotation, c.Body.BaseComLinearVelocity);
            var cmd = c.Command;
            speedValue.text = $"{v.x,5:0.00} / {cmd.x,4:0.00} m/s";
            speedActual.style.width = Length.Percent(Mathf.Clamp01(Mathf.Abs(v.x)) * 100f);
            speedCmd.style.width = Length.Percent(Mathf.Clamp01(Mathf.Abs(cmd.x)) * 100f);
            heightValue.text = $"{c.BaseHeight:0.00} m";
            var tilt = Vector3.Angle(c.Body.BaseRotation * Vector3.up, Vector3.up);
            tiltValue.text = $"{tilt:0} °";
            tiltValue.EnableInClassList("alert", tilt > 30f);
            var b = c.Body;
            payloadValue.text = b.AddedMass == 0f ? "none" : $"{b.AddedMass:+0.0;-0.0} kg";
            manager.Limits.Clamp(Quantity.Payload, b.AddedMass, false, out var outside);
            payloadValue.EnableInClassList("alert", outside);
            if (c.FootFriction != null)
                for (var i = 0; i < 4; i++)
                    feet.SetFriction(i, c.FootFriction.EffectiveStatic(i));
        }

        void RefreshEvents()
        {
            var events = manager.Log.Events;
            if (events.Count == shownEvents)
                return;
            shownEvents = events.Count;
            eventsList.Clear();
            for (var i = events.Count - 1; i >= Mathf.Max(0, events.Count - 8); i--)
            {
                var e = events[i];
                var row = new VisualElement { pickingMode = PickingMode.Ignore };
                row.AddToClassList("event");
                row.AddToClassList(e.Type);
                row.EnableInClassList("outside", e.OutsideTraining);
                var bar = new VisualElement { pickingMode = PickingMode.Ignore };
                bar.AddToClassList("event-bar");
                var time = new Label(e.Time.ToString("0.0", CultureInfo.InvariantCulture) + " s") { pickingMode = PickingMode.Ignore };
                time.AddToClassList("event-time");
                time.AddToClassList("num");
                var text = new Label($"{e.Type}: {e.Detail}") { pickingMode = PickingMode.Ignore };
                text.AddToClassList("event-text");
                row.Add(bar);
                row.Add(time);
                row.Add(text);
                eventsList.Add(row);
            }
            UiFonts.ApplyMono(eventsList);
        }

        void RefreshDrive()
        {
            if (commands == null)
                return;
            fixedButton.EnableInClassList("on", commands.UseFixedCommand);
            manualButton.EnableInClassList("on", !commands.UseFixedCommand);
            cruise.style.display = commands.UseFixedCommand ? DisplayStyle.Flex : DisplayStyle.None;
            driveDetail.Q("manual-hint").style.display = commands.UseFixedCommand ? DisplayStyle.None : DisplayStyle.Flex;
            cruise.Refresh();
        }
    }
}
