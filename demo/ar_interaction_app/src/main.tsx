import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AlertTriangle,
  BarChart3,
  ChevronLeft,
  ChevronRight,
  CheckCircle2,
  Circle,
  Crosshair,
  MousePointer2,
  Play,
  Radio,
  RotateCcw,
  ScanLine,
  Settings2,
  Sparkles,
  Timer,
  Video,
  WifiOff,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import * as THREE from "three";
import "./styles.css";

type PageId = "demo" | "guide" | "results" | "charts";
type MethodId = "c1t_tcn" | "c6_ensemble";
type SourceId = "replay" | "webcam";
type TaskId =
  | "object"
  | "carousel"
  | "scroll"
  | "browser"
  | "transfer"
  | "targets"
  | "placement"
  | "inspection"
  | "measurement"
  | "assembly"
  | "info"
  | "docking"
  | "tour";
type InteractionMode = "direct" | "c4_task_aware";
type BackendHealth = "checking" | "ready" | "offline";
type GestureId =
  | "no_gesture"
  | "point_2f"
  | "click_2f"
  | "swipe_left"
  | "swipe_right"
  | "zoom_in"
  | "zoom_out";

type ActionId =
  | "idle"
  | "pointer_hover"
  | "select_confirm"
  | "navigate_previous"
  | "navigate_next"
  | "zoom_in"
  | "zoom_out";

type SceneState = {
  rotationY: number;
  scale: number;
  selected: boolean;
  anchored: boolean;
  pointerX: number;
  pointerY: number;
  action: ActionId;
  carouselIndex: number;
  targetIndex: number;
  scrollIndex: number;
  browserIndex: number;
  transferIndex: number;
  transferHeld: boolean;
  hits: number;
  inspectionPanel: number;
};

type TaskStep = {
  id: string;
  action: ActionId;
  gesture: GestureId;
  label: string;
};

type TaskDefinition = {
  id: TaskId;
  label: string;
  description: string;
  steps: TaskStep[];
};

type TaskRunState = {
  active: boolean;
  completed: number;
  mistakes: number;
  startedAt: number | null;
  elapsedMs: number;
  lastAcceptedAt: number;
  lastEvaluatedAt: number;
  lastAcceptedAction: ActionId | "idle";
};

type StreamMessage = {
  type: "prediction" | "error" | "status";
  gesture?: GestureId;
  confidence?: number;
  action?: ActionId;
  event?: {
    action: ActionId;
    gesture: GestureId;
    confidence: number;
    state: string;
    timestamp_ms: number;
  } | null;
  message?: string;
  source?: SourceId;
  sample_id?: string;
  target_label?: string;
  detection_rate?: number;
  preview_image?: string;
  landmarks?: number[][];
  pointer?: { x: number; y: number } | null;
  fps?: number | null;
  processing_ms?: number | null;
  camera?: CameraStats | null;
  session_id?: string;
  log_path?: string;
  policy_context?: {
    mode?: string;
    step_index?: number;
    step_count?: number;
    expected_action?: ActionId | "";
    expected_label?: GestureId | "";
    expected_id?: string;
    false_events?: number;
  } | null;
};

type PolicyContext = NonNullable<StreamMessage["policy_context"]>;

type CameraStats = {
  running?: boolean;
  error?: string;
  camera_index?: number;
  requested_width?: number;
  requested_height?: number;
  target_fps?: number;
  width?: number;
  height?: number;
  capture_fps?: number;
  frame_age_ms?: number | null;
  backend?: string;
};

const BACKEND_HTTP_URL = "http://127.0.0.1:8000";
const BACKEND_WS_URL = "ws://127.0.0.1:8000";
const FAST_CAMERA_WIDTH = 1280;
const FAST_CAMERA_HEIGHT = 720;
const DEFAULT_TARGET_FPS = 30;
const DEFAULT_INTERVAL_MS = Math.round(1000 / DEFAULT_TARGET_FPS);
const DEFAULT_PREVIEW_WIDTH = 960;
const DEFAULT_JPEG_QUALITY = 84;
const POINTER_SMOOTHING = 0.34;

const methods: Array<{
  id: MethodId;
  label: string;
  artifact: string;
  accuracy: string;
  macroF1: string;
  latency: string;
  role: string;
}> = [
  {
    id: "c1t_tcn",
    label: "Baseline TCN",
    artifact: "validated temporal TCN",
    accuracy: "0.907",
    macroF1: "0.850",
    latency: "4.64 ms p95",
    role: "direct control baseline"
  },
  {
    id: "c6_ensemble",
    label: "Robust C6",
    artifact: "validated + augmented TCN",
    accuracy: "0.930",
    macroF1: "0.887",
    latency: "4.34 ms p95",
    role: "recognition upgrade"
  }
];

const tasks: TaskDefinition[] = [
  {
    id: "object",
    label: "Object: select and scale",
    description: "Point at the cube, confirm once, then change its size.",
    steps: [
      { id: "object_hover", action: "pointer_hover", gesture: "point_2f", label: "Point at cube" },
      { id: "object_select", action: "select_confirm", gesture: "click_2f", label: "Short click" },
      { id: "object_zoom_in", action: "zoom_in", gesture: "zoom_in", label: "Make bigger" },
      { id: "object_zoom_out", action: "zoom_out", gesture: "zoom_out", label: "Make smaller" }
    ]
  },
  {
    id: "carousel",
    label: "Gallery Navigation",
    description: "Move through virtual items and confirm selection.",
    steps: [
      { id: "carousel_next_1", action: "navigate_next", gesture: "swipe_right", label: "Next" },
      { id: "carousel_next_2", action: "navigate_next", gesture: "swipe_right", label: "Next" },
      { id: "carousel_previous", action: "navigate_previous", gesture: "swipe_left", label: "Previous" },
      { id: "carousel_select", action: "select_confirm", gesture: "click_2f", label: "Confirm" }
    ]
  },
  {
    id: "scroll",
    label: "List: scroll and open",
    description: "Swipe through a floating list, then open the highlighted row.",
    steps: [
      { id: "scroll_next_1", action: "navigate_next", gesture: "swipe_right", label: "Scroll down" },
      { id: "scroll_next_2", action: "navigate_next", gesture: "swipe_right", label: "Scroll down again" },
      { id: "scroll_previous", action: "navigate_previous", gesture: "swipe_left", label: "Scroll up" },
      { id: "scroll_select", action: "select_confirm", gesture: "click_2f", label: "Short click row" }
    ]
  },
  {
    id: "browser",
    label: "Cards: browse and inspect",
    description: "Point at the card carousel, move to the next cards, then inspect one.",
    steps: [
      { id: "browser_point", action: "pointer_hover", gesture: "point_2f", label: "Point at cards" },
      { id: "browser_next_1", action: "navigate_next", gesture: "swipe_right", label: "Next card" },
      { id: "browser_next_2", action: "navigate_next", gesture: "swipe_right", label: "Next card again" },
      { id: "browser_open", action: "select_confirm", gesture: "click_2f", label: "Short click open" },
      { id: "browser_zoom", action: "zoom_in", gesture: "zoom_in", label: "Zoom to inspect" }
    ]
  },
  {
    id: "transfer",
    label: "Sorting: move item",
    description: "Pick one virtual item, move it to the right bin, and drop it.",
    steps: [
      { id: "transfer_point", action: "pointer_hover", gesture: "point_2f", label: "Point at item" },
      { id: "transfer_pick", action: "select_confirm", gesture: "click_2f", label: "Short click pick" },
      { id: "transfer_move", action: "navigate_next", gesture: "swipe_right", label: "Move to right bin" },
      { id: "transfer_drop", action: "select_confirm", gesture: "click_2f", label: "Short click drop" },
      { id: "transfer_return", action: "navigate_previous", gesture: "swipe_left", label: "Return left" }
    ]
  },
  {
    id: "targets",
    label: "Target Selection",
    description: "Hover and confirm targets in an AR scene.",
    steps: [
      { id: "target_hover_1", action: "pointer_hover", gesture: "point_2f", label: "Hover A" },
      { id: "target_confirm_1", action: "select_confirm", gesture: "click_2f", label: "Confirm A" },
      { id: "target_hover_2", action: "pointer_hover", gesture: "point_2f", label: "Hover B" },
      { id: "target_confirm_2", action: "select_confirm", gesture: "click_2f", label: "Confirm B" }
    ]
  },
  {
    id: "placement",
    label: "Object Placement",
    description: "Place an object on a detected surface and adjust its size.",
    steps: [
      { id: "place_aim", action: "pointer_hover", gesture: "point_2f", label: "Aim" },
      { id: "place_anchor", action: "select_confirm", gesture: "click_2f", label: "Anchor" },
      { id: "place_zoom", action: "zoom_in", gesture: "zoom_in", label: "Fit" },
      { id: "place_rotate", action: "navigate_next", gesture: "swipe_right", label: "Orient" }
    ]
  },
  {
    id: "inspection",
    label: "Object Inspection",
    description: "Select, rotate, inspect detail states, and return scale.",
    steps: [
      { id: "inspect_select", action: "select_confirm", gesture: "click_2f", label: "Select" },
      { id: "inspect_rotate_next", action: "navigate_next", gesture: "swipe_right", label: "Rotate" },
      { id: "inspect_zoom", action: "zoom_in", gesture: "zoom_in", label: "Inspect" },
      { id: "inspect_rotate_back", action: "navigate_previous", gesture: "swipe_left", label: "Back" },
      { id: "inspect_zoom_out", action: "zoom_out", gesture: "zoom_out", label: "Reset" }
    ]
  },
  {
    id: "measurement",
    label: "Distance Measure",
    description: "Mark two AR points and magnify the measured segment.",
    steps: [
      { id: "measure_aim_a", action: "pointer_hover", gesture: "point_2f", label: "Aim A" },
      { id: "measure_pin_a", action: "select_confirm", gesture: "click_2f", label: "Pin A" },
      { id: "measure_aim_b", action: "pointer_hover", gesture: "point_2f", label: "Aim B" },
      { id: "measure_pin_b", action: "select_confirm", gesture: "click_2f", label: "Pin B" },
      { id: "measure_zoom", action: "zoom_in", gesture: "zoom_in", label: "Magnify" }
    ]
  },
  {
    id: "assembly",
    label: "Assembly Assist",
    description: "Pick a virtual part, move to the next slot, and lock it.",
    steps: [
      { id: "assembly_point_part", action: "pointer_hover", gesture: "point_2f", label: "Point Part" },
      { id: "assembly_pick", action: "select_confirm", gesture: "click_2f", label: "Pick" },
      { id: "assembly_next_slot", action: "navigate_next", gesture: "swipe_right", label: "Next Slot" },
      { id: "assembly_align", action: "pointer_hover", gesture: "point_2f", label: "Align" },
      { id: "assembly_lock", action: "select_confirm", gesture: "click_2f", label: "Lock" }
    ]
  },
  {
    id: "info",
    label: "Info Panel",
    description: "Open an object card and browse AR metadata panels.",
    steps: [
      { id: "info_point", action: "pointer_hover", gesture: "point_2f", label: "Point" },
      { id: "info_open", action: "select_confirm", gesture: "click_2f", label: "Open" },
      { id: "info_next", action: "navigate_next", gesture: "swipe_right", label: "Next Card" },
      { id: "info_prev", action: "navigate_previous", gesture: "swipe_left", label: "Previous" },
      { id: "info_close", action: "select_confirm", gesture: "click_2f", label: "Close" }
    ]
  },
  {
    id: "docking",
    label: "Precision Docking",
    description: "Align the AR object with a target reticle and dock it.",
    steps: [
      { id: "dock_pointer", action: "pointer_hover", gesture: "point_2f", label: "Acquire" },
      { id: "dock_left", action: "navigate_previous", gesture: "swipe_left", label: "Nudge Left" },
      { id: "dock_right", action: "navigate_next", gesture: "swipe_right", label: "Nudge Right" },
      { id: "dock_zoom", action: "zoom_out", gesture: "zoom_out", label: "Fine Scale" },
      { id: "dock_confirm", action: "select_confirm", gesture: "click_2f", label: "Dock" }
    ]
  },
  {
    id: "tour",
    label: "Guided Tour",
    description: "Navigate scene waypoints, focus one item, then return.",
    steps: [
      { id: "tour_next_1", action: "navigate_next", gesture: "swipe_right", label: "Waypoint 1" },
      { id: "tour_select", action: "select_confirm", gesture: "click_2f", label: "Focus" },
      { id: "tour_zoom", action: "zoom_in", gesture: "zoom_in", label: "Zoom" },
      { id: "tour_next_2", action: "navigate_next", gesture: "swipe_right", label: "Waypoint 2" },
      { id: "tour_back", action: "navigate_previous", gesture: "swipe_left", label: "Return" }
    ]
  }
];

const liveTaskIds: TaskId[] = ["object", "scroll", "browser", "transfer"];
const liveTasks = tasks.filter((item) => liveTaskIds.includes(item.id));

const actionLabels: Record<ActionId, string> = {
  idle: "Idle",
  pointer_hover: "Pointer",
  select_confirm: "Select",
  navigate_previous: "Previous",
  navigate_next: "Next",
  zoom_in: "Zoom In",
  zoom_out: "Zoom Out"
};

const interactionModeLabels: Record<InteractionMode, string> = {
  direct: "Direct",
  c4_task_aware: "TARC"
};

const researchStages = [
  {
    title: "Official Methods",
    value: "3",
    detail: "baseline, robust recognizer, proposed controller"
  },
  {
    title: "Recognition Gain",
    value: "+0.037",
    detail: "macro F1 from M1 to M2"
  },
  {
    title: "False Cost",
    value: "-77%",
    detail: "M3 vs baseline direct"
  },
  {
    title: "Task Success",
    value: "0.531",
    detail: "M3 maintains baseline-level completion"
  }
];

const c4TaskBenchmarkResults = {
  directSuccess: "0.527",
  c3c2Success: "0.552",
  taskAwareSuccess: "0.531",
  directFalseCost: "0.110",
  c3c2FalseCost: "0.085",
  taskAwareFalseCost: "0.025",
  taskAwarePrecision: "0.974",
  taskAwareUnintended: "0.024"
};

const c4TaskRows = [
  {
    method: "M1 Baseline Direct",
    success: "0.527",
    precision: "0.892",
    recall: "0.860",
    unintended: "0.106",
    falseCost: "0.110"
  },
  {
    method: "M2 Robust Direct",
    success: "0.552",
    precision: "0.917",
    recall: "0.866",
    unintended: "0.081",
    falseCost: "0.085"
  },
  {
    method: "M3 Proposed TARC",
    success: "0.531",
    precision: "0.974",
    recall: "0.857",
    unintended: "0.024",
    falseCost: "0.025"
  }
];

const recognitionChartRows = [
  { method: "M1 Baseline TCN", accuracy: 0.907, macroF1: 0.85 },
  { method: "M2 Robust C6", accuracy: 0.93, macroF1: 0.887 }
];

const c4RiskChartRows = [
  { method: "M1 Baseline Direct", precision: 0.892, recall: 0.86, unintended: 0.106, falseCost: 0.11 },
  { method: "M2 Robust Direct", precision: 0.917, recall: 0.866, unintended: 0.081, falseCost: 0.085 },
  { method: "M3 Proposed TARC", precision: 0.974, recall: 0.857, unintended: 0.024, falseCost: 0.025 }
];

const c4TaskChartRows = [
  { method: "M1 Baseline Direct", success: 0.527, precision: 0.892, recall: 0.86, unintended: 0.106, falseCost: 0.11 },
  { method: "M2 Robust Direct", success: 0.552, precision: 0.917, recall: 0.866, unintended: 0.081, falseCost: 0.085 },
  { method: "M3 Proposed TARC", success: 0.531, precision: 0.974, recall: 0.857, unintended: 0.024, falseCost: 0.025 }
];

const weakTaskRows = [
  { task: "inspection", success: 0.175, recall: 0.755, falseCost: 0.035, missedCost: 0.264 },
  { task: "info", success: 0.3, recall: 0.825, falseCost: 0.08, missedCost: 0.202 },
  { task: "transfer", success: 0.3, recall: 0.795, falseCost: 0.037, missedCost: 0.248 },
  { task: "tour", success: 0.3, recall: 0.82, falseCost: 0.033, missedCost: 0.179 },
  { task: "docking", success: 0.35, recall: 0.825, falseCost: 0.042, missedCost: 0.191 },
  { task: "assembly", success: 0.375, recall: 0.825, falseCost: 0.06, missedCost: 0.217 }
];

const gestureActions: Record<GestureId, ActionId> = {
  no_gesture: "idle",
  point_2f: "pointer_hover",
  click_2f: "select_confirm",
  swipe_left: "navigate_previous",
  swipe_right: "navigate_next",
  zoom_in: "zoom_in",
  zoom_out: "zoom_out"
};

const gestureControls: Array<{
  id: GestureId;
  label: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
}> = [
  { id: "point_2f", label: "Point", icon: MousePointer2 },
  { id: "click_2f", label: "Click", icon: Crosshair },
  { id: "swipe_left", label: "Left", icon: ChevronLeft },
  { id: "swipe_right", label: "Right", icon: ChevronRight },
  { id: "zoom_in", label: "In", icon: ZoomIn },
  { id: "zoom_out", label: "Out", icon: ZoomOut }
];

const gestureGuideCards: Array<{
  id: GestureId;
  label: string;
  action: string;
  pose: string;
  cue: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
}> = [
  {
    id: "point_2f",
    label: "Point",
    action: "Move cursor",
    pose: "Hold index and middle fingers visible.",
    cue: "Keep the hand steady before selecting.",
    icon: MousePointer2
  },
  {
    id: "click_2f",
    label: "Click",
    action: "Confirm",
    pose: "Briefly bring index and middle fingers close.",
    cue: "Do not hold this gesture; release back to Point.",
    icon: Crosshair
  },
  {
    id: "swipe_left",
    label: "Swipe Left",
    action: "Previous",
    pose: "Move the visible hand horizontally left.",
    cue: "Use one clean motion, then pause.",
    icon: ChevronLeft
  },
  {
    id: "swipe_right",
    label: "Swipe Right",
    action: "Next",
    pose: "Move the visible hand horizontally right.",
    cue: "Keep the motion wider than a small tremor.",
    icon: ChevronRight
  },
  {
    id: "zoom_in",
    label: "Zoom In",
    action: "Scale up",
    pose: "Open the finger distance during the gesture.",
    cue: "Start small, then open clearly.",
    icon: ZoomIn
  },
  {
    id: "zoom_out",
    label: "Zoom Out",
    action: "Scale down",
    pose: "Close the finger distance during the gesture.",
    cue: "Start open, then close clearly.",
    icon: ZoomOut
  }
];

function initialSceneState(): SceneState {
  return {
    rotationY: 0,
    scale: 0.68,
    selected: false,
    anchored: false,
    pointerX: 0,
    pointerY: 0,
    action: "idle",
    carouselIndex: 0,
    targetIndex: 0,
    scrollIndex: 0,
    browserIndex: 0,
    transferIndex: 0,
    transferHeld: false,
    hits: 0,
    inspectionPanel: 0
  };
}

function initialTaskRunState(): TaskRunState {
  return {
    active: false,
    completed: 0,
    mistakes: 0,
    startedAt: null,
    elapsedMs: 0,
    lastAcceptedAt: 0,
    lastEvaluatedAt: 0,
    lastAcceptedAction: "idle"
  };
}

function applyAction(state: SceneState, action: ActionId, task: TaskId): SceneState {
  if (action === "navigate_previous") {
    return {
      ...state,
      rotationY: state.rotationY - Math.PI / 5,
      carouselIndex: Math.max(0, state.carouselIndex - 1),
      targetIndex: Math.max(0, state.targetIndex - 1),
      scrollIndex: Math.max(0, state.scrollIndex - 1),
      browserIndex: Math.max(0, state.browserIndex - 1),
      transferIndex: Math.max(0, state.transferIndex - 1),
      inspectionPanel: Math.max(0, state.inspectionPanel - 1),
      action
    };
  }
  if (action === "navigate_next") {
    return {
      ...state,
      rotationY: state.rotationY + Math.PI / 5,
      carouselIndex: Math.min(4, state.carouselIndex + 1),
      targetIndex: Math.min(4, state.targetIndex + 1),
      scrollIndex: Math.min(5, state.scrollIndex + 1),
      browserIndex: Math.min(4, state.browserIndex + 1),
      transferIndex: Math.min(3, state.transferIndex + 1),
      inspectionPanel: Math.min(3, state.inspectionPanel + 1),
      action
    };
  }
  if (action === "zoom_in") {
    return { ...state, scale: Math.min(1.36, state.scale + 0.1), action };
  }
  if (action === "zoom_out") {
    return { ...state, scale: Math.max(0.52, state.scale - 0.1), action };
  }
  if (action === "select_confirm") {
    const latchSelectionTasks: TaskId[] = ["targets", "measurement", "docking", "scroll", "browser"];
    const hitTasks: TaskId[] = [
      "targets",
      "placement",
      "measurement",
      "assembly",
      "info",
      "docking",
      "tour",
      "scroll",
      "browser",
      "transfer"
    ];
    return {
      ...state,
      selected: latchSelectionTasks.includes(task) ? true : !state.selected,
      anchored: task === "placement" || task === "docking" ? true : state.anchored,
      transferHeld: task === "transfer" ? !state.transferHeld : state.transferHeld,
      hits: hitTasks.includes(task) ? state.hits + 1 : state.hits,
      action
    };
  }
  if (action === "pointer_hover") {
    return {
      ...state,
      pointerX: Math.max(-0.92, Math.min(0.92, state.pointerX + 0.16)),
      pointerY: Math.sin(Date.now() / 420) * 0.3,
      action
    };
  }
  return { ...state, action: "idle" };
}

function applyGesture(state: SceneState, gesture: GestureId, task: TaskId): SceneState {
  return applyAction(state, gestureActions[gesture], task);
}

function advanceTaskRun(current: TaskRunState, taskDefinition: TaskDefinition, action: ActionId): TaskRunState {
  if (!current.active || action === "idle" || current.completed >= taskDefinition.steps.length) {
    return current;
  }

  const now = Date.now();
  if (now - current.lastEvaluatedAt < 420) {
    return current;
  }

  const expected = taskDefinition.steps[current.completed];
  if (action === expected.action) {
    const completed = current.completed + 1;
    return {
      ...current,
      active: completed < taskDefinition.steps.length,
      completed,
      elapsedMs: current.startedAt ? now - current.startedAt : current.elapsedMs,
      lastAcceptedAt: now,
      lastEvaluatedAt: now,
      lastAcceptedAction: action
    };
  }

  return {
    ...current,
    mistakes: current.mistakes + 1,
    lastEvaluatedAt: now
  };
}

function SceneCanvas({ state, task }: { state: SceneState; task: TaskId }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const objectRef = useRef<THREE.Group | null>(null);
  const pointerRef = useRef<THREE.Mesh | null>(null);
  const targetGroupRef = useRef<THREE.Group | null>(null);
  const scrollGroupRef = useRef<THREE.Group | null>(null);
  const browserGroupRef = useRef<THREE.Group | null>(null);
  const transferGroupRef = useRef<THREE.Group | null>(null);
  const anchorRef = useRef<THREE.Mesh | null>(null);
  const materialRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const stateRef = useRef(state);
  const taskRef = useRef(task);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    taskRef.current = task;
  }, [task]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 1.35, 5.4);

    const key = new THREE.DirectionalLight(0xe6f7ff, 2.6);
    key.position.set(3, 4, 5);
    scene.add(key);
    scene.add(new THREE.HemisphereLight(0x9ed4ff, 0x2d3328, 1.4));

    const grid = new THREE.GridHelper(8, 16, 0x51606a, 0x253039);
    grid.position.y = -1.35;
    scene.add(grid);

    const group = new THREE.Group();
    group.position.y = 0.48;
    const material = new THREE.MeshStandardMaterial({
      color: 0x72d3c9,
      metalness: 0.26,
      roughness: 0.28,
      emissive: 0x061f1d,
      emissiveIntensity: 0.35
    });
    materialRef.current = material;
    const cube = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.9, 0.9), material);
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.86, 0.019, 16, 96),
      new THREE.MeshStandardMaterial({ color: 0xf2cc60, roughness: 0.4, metalness: 0.1 })
    );
    ring.rotation.x = Math.PI / 2;
    group.add(cube, ring);
    scene.add(group);
    objectRef.current = group;

    const targetGroup = new THREE.Group();
    for (let index = 0; index < 5; index += 1) {
      const target = new THREE.Mesh(
        new THREE.SphereGeometry(0.1, 24, 24),
        new THREE.MeshStandardMaterial({ color: 0xd7b95c, roughness: 0.35, metalness: 0.2 })
      );
      target.position.set((index - 2) * 0.58, -0.58, 1.2);
      targetGroup.add(target);
    }
    scene.add(targetGroup);
    targetGroupRef.current = targetGroup;

    const scrollGroup = new THREE.Group();
    scrollGroup.position.set(-0.05, 0.08, 0.42);
    for (let index = 0; index < 6; index += 1) {
      const row = new THREE.Group();
      row.userData.baseY = 0.78 - index * 0.31;
      const rowBack = new THREE.Mesh(
        new THREE.BoxGeometry(2.18, 0.22, 0.035),
        new THREE.MeshStandardMaterial({
          color: index % 2 === 0 ? 0x24323a : 0x1b272d,
          roughness: 0.48,
          metalness: 0.08,
          transparent: true,
          opacity: 0.88
        })
      );
      const marker = new THREE.Mesh(
        new THREE.BoxGeometry(0.22, 0.12, 0.045),
        new THREE.MeshStandardMaterial({ color: 0x72d3c9, roughness: 0.35, metalness: 0.12 })
      );
      const progress = new THREE.Mesh(
        new THREE.BoxGeometry(0.8 - index * 0.055, 0.035, 0.05),
        new THREE.MeshStandardMaterial({ color: 0xd7b95c, roughness: 0.4, metalness: 0.05 })
      );
      marker.position.x = -0.88;
      progress.position.x = -0.18;
      row.add(rowBack, marker, progress);
      row.position.y = row.userData.baseY;
      scrollGroup.add(row);
    }
    scene.add(scrollGroup);
    scrollGroupRef.current = scrollGroup;

    const browserGroup = new THREE.Group();
    browserGroup.position.set(0, 0.22, 0.18);
    for (let index = 0; index < 5; index += 1) {
      const card = new THREE.Group();
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(0.92, 1.08, 0.045),
        new THREE.MeshStandardMaterial({
          color: [0x263e4d, 0x31432e, 0x3f344e, 0x4a3f2d, 0x2f4542][index],
          roughness: 0.42,
          metalness: 0.08,
          transparent: true,
          opacity: 0.92
        })
      );
      const header = new THREE.Mesh(
        new THREE.BoxGeometry(0.72, 0.09, 0.052),
        new THREE.MeshStandardMaterial({ color: 0xf2cc60, roughness: 0.35, metalness: 0.08 })
      );
      const thumb = new THREE.Mesh(
        new THREE.BoxGeometry(0.44, 0.36, 0.056),
        new THREE.MeshStandardMaterial({ color: 0x72d3c9, roughness: 0.32, metalness: 0.14 })
      );
      header.position.y = 0.36;
      thumb.position.y = -0.03;
      card.add(panel, header, thumb);
      card.userData.index = index;
      browserGroup.add(card);
    }
    scene.add(browserGroup);
    browserGroupRef.current = browserGroup;

    const transferGroup = new THREE.Group();
    transferGroup.position.set(0, 1.7, 0.38);
    const trayMaterial = new THREE.MeshStandardMaterial({
      color: 0x243039,
      roughness: 0.5,
      metalness: 0.05,
      transparent: true,
      opacity: 0.86
    });
    const leftTray = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.16, 0.7), trayMaterial);
    const rightTray = new THREE.Mesh(new THREE.BoxGeometry(1.2, 0.16, 0.7), trayMaterial.clone());
    leftTray.position.set(-0.78, -0.76, 0);
    rightTray.position.set(0.78, -0.76, 0);
    transferGroup.add(leftTray, rightTray);
    for (let index = 0; index < 4; index += 1) {
      const item = new THREE.Mesh(
        new THREE.BoxGeometry(0.36, 0.36, 0.36),
        new THREE.MeshStandardMaterial({
          color: [0x72d3c9, 0xd7b95c, 0xff7a72, 0x9cf18b][index],
          roughness: 0.34,
          metalness: 0.18
        })
      );
      item.userData.left = new THREE.Vector3(-1.06 + index * 0.28, -0.42, 0.12);
      item.userData.right = new THREE.Vector3(0.44 + index * 0.28, -0.42, 0.12);
      item.position.copy(item.userData.left);
      transferGroup.add(item);
    }
    scene.add(transferGroup);
    transferGroupRef.current = transferGroup;

    const anchor = new THREE.Mesh(
      new THREE.TorusGeometry(0.42, 0.015, 12, 80),
      new THREE.MeshStandardMaterial({ color: 0x9cf18b, roughness: 0.42, metalness: 0.08 })
    );
    anchor.rotation.x = Math.PI / 2;
    anchor.position.set(0, -1.18, 0.72);
    scene.add(anchor);
    anchorRef.current = anchor;

    const pointer = new THREE.Mesh(
      new THREE.SphereGeometry(0.055, 24, 24),
      new THREE.MeshStandardMaterial({ color: 0xff6b6b, emissive: 0x3a0505, emissiveIntensity: 1.2 })
    );
    pointer.position.set(0, 0.62, 1.22);
    scene.add(pointer);
    pointerRef.current = pointer;

    let frameId = 0;
    const clock = new THREE.Clock();

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    const animate = () => {
      const current = stateRef.current;
      const currentTask = taskRef.current;
      const elapsed = clock.getElapsedTime();

      if (objectRef.current) {
        objectRef.current.visible = currentTask !== "scroll" && currentTask !== "browser" && currentTask !== "transfer";
        const targetX =
          currentTask === "carousel"
            ? (current.carouselIndex - 2) * -0.42
            : currentTask === "placement" || currentTask === "measurement" || currentTask === "assembly" || currentTask === "docking"
              ? current.pointerX * 1.35
              : 0;
        const targetY = currentTask === "placement" || currentTask === "docking" ? -0.1 : 0.48;
        objectRef.current.position.x += (targetX - objectRef.current.position.x) * 0.08;
        objectRef.current.position.y += (targetY - objectRef.current.position.y) * 0.08;
        objectRef.current.rotation.x = Math.sin(elapsed * 0.55) * 0.08;
        objectRef.current.rotation.y += (current.rotationY - objectRef.current.rotation.y) * 0.08;
        const targetScale = current.selected || current.anchored ? current.scale * 1.04 : current.scale;
        objectRef.current.scale.lerp(new THREE.Vector3(targetScale, targetScale, targetScale), 0.08);
      }

      if (targetGroupRef.current) {
        targetGroupRef.current.visible =
          currentTask === "targets" ||
          currentTask === "carousel" ||
          currentTask === "inspection" ||
          currentTask === "assembly" ||
          currentTask === "info" ||
          currentTask === "tour";
        targetGroupRef.current.children.forEach((child, index) => {
          const activeIndex =
            currentTask === "inspection" || currentTask === "info"
              ? current.inspectionPanel
              : currentTask === "carousel" || currentTask === "tour" || currentTask === "assembly"
                ? current.carouselIndex
                : current.targetIndex;
          const scale = index === activeIndex ? 1.55 : 1;
          child.scale.lerp(new THREE.Vector3(scale, scale, scale), 0.12);
        });
      }

      if (scrollGroupRef.current) {
        scrollGroupRef.current.visible = currentTask === "scroll";
        scrollGroupRef.current.children.forEach((child, index) => {
          const row = child as THREE.Group;
          const active = index === current.scrollIndex;
          const rowTargetY = Number(row.userData.baseY ?? 0) + current.scrollIndex * 0.18;
          row.position.y += (rowTargetY - row.position.y) * 0.1;
          row.position.z = active ? 0.12 : 0;
          const scaleX = active || (current.selected && index === current.scrollIndex) ? 1.08 : 1;
          const scaleY = active ? 1.12 : 1;
          row.scale.lerp(new THREE.Vector3(scaleX, scaleY, 1), 0.14);
          row.children.forEach((part) => {
            const mesh = part as THREE.Mesh;
            const material = mesh.material as THREE.MeshStandardMaterial;
            if (material?.emissive) {
              material.emissive.set(active ? 0x102d29 : 0x000000);
              material.emissiveIntensity = active ? 0.55 : 0;
            }
          });
        });
      }

      if (browserGroupRef.current) {
        browserGroupRef.current.visible = currentTask === "browser";
        browserGroupRef.current.children.forEach((child, index) => {
          const card = child as THREE.Group;
          const offset = index - current.browserIndex;
          const targetX = offset * 0.74;
          const targetZ = -Math.abs(offset) * 0.18;
          card.position.x += (targetX - card.position.x) * 0.1;
          card.position.z += (targetZ - card.position.z) * 0.1;
          card.rotation.y += (-offset * 0.18 - card.rotation.y) * 0.1;
          const activeScale = index === current.browserIndex ? (current.selected ? current.scale * 1.1 : 1.08) : 0.74;
          card.scale.lerp(new THREE.Vector3(activeScale, activeScale, activeScale), 0.12);
        });
      }

      if (transferGroupRef.current) {
        transferGroupRef.current.visible = currentTask === "transfer";
        const movedItems = Math.min(4, Math.floor(current.hits / 2));
        transferGroupRef.current.children.forEach((child, index) => {
          if (index < 2) {
            const tray = child as THREE.Mesh;
            const activeTray = current.transferIndex > 0 ? index === 1 : index === 0;
            const trayScale = activeTray ? 1.08 : 1;
            tray.scale.lerp(new THREE.Vector3(trayScale, 1, trayScale), 0.12);
            return;
          }
          const itemIndex = index - 2;
          const item = child as THREE.Mesh;
          const left = item.userData.left as THREE.Vector3;
          const right = item.userData.right as THREE.Vector3;
          const held = current.transferHeld && itemIndex === movedItems;
          const target = held
            ? new THREE.Vector3(current.transferIndex > 0 ? 1.12 : -0.28 + current.pointerX, current.pointerY + 0.04, 0.48)
            : itemIndex < movedItems
              ? right
              : left;
          item.position.lerp(target, 0.13);
          item.rotation.y += held ? 0.08 : 0.025;
          const activeScale = held || itemIndex === Math.min(3, movedItems) ? 1.18 : 1;
          item.scale.lerp(new THREE.Vector3(activeScale, activeScale, activeScale), 0.12);
        });
      }

      if (anchorRef.current) {
        anchorRef.current.visible = currentTask === "placement" || currentTask === "measurement" || currentTask === "docking";
        anchorRef.current.position.x += (current.pointerX * 1.35 - anchorRef.current.position.x) * 0.12;
        const anchorScale = current.anchored ? 1.3 : 1 + Math.sin(elapsed * 4) * 0.08;
        anchorRef.current.scale.lerp(new THREE.Vector3(anchorScale, anchorScale, anchorScale), 0.14);
      }

      if (pointerRef.current) {
        pointerRef.current.position.x += (current.pointerX - pointerRef.current.position.x) * 0.18;
        pointerRef.current.position.y += (current.pointerY + 0.68 - pointerRef.current.position.y) * 0.18;
        pointerRef.current.visible = current.action === "pointer_hover" || current.selected || currentTask === "targets";
      }

      if (materialRef.current) {
        materialRef.current.color.set(current.selected || current.anchored ? 0x9cf18b : 0x72d3c9);
        materialRef.current.emissive.set(current.selected || current.anchored ? 0x183a10 : 0x061f1d);
      }

      renderer.render(scene, camera);
      frameId = requestAnimationFrame(animate);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameId);
      observer.disconnect();
      renderer.dispose();
      container.removeChild(renderer.domElement);
    };
  }, []);

  return <div className="scene" ref={containerRef} data-testid="ar-scene" />;
}

function MetricBarChart({
  title,
  subtitle,
  values,
  valueKey,
  lowerIsBetter = false
}: {
  title: string;
  subtitle: string;
  values: Array<Record<string, string | number>>;
  valueKey: string;
  lowerIsBetter?: boolean;
}) {
  const max = Math.max(...values.map((item) => Number(item[valueKey])), 0.01);

  return (
    <section className={`panel chart-panel ${lowerIsBetter ? "risk-chart" : ""}`}>
      <div className="chart-heading">
        <h3>{title}</h3>
        <p>{subtitle}</p>
      </div>
      <div className="bar-chart" aria-label={title}>
        {values.map((item) => {
          const value = Number(item[valueKey]);
          const width = `${Math.max(2, (value / max) * 100)}%`;
          return (
            <div className="bar-row" key={`${item.method}-${valueKey}`}>
              <div className="bar-label">
                <span>{item.method}</span>
                <strong>{value.toFixed(3)}</strong>
              </div>
              <div className="bar-track">
                <span style={{ width }} />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TradeoffChart() {
  const maxCost = Math.max(...c4TaskChartRows.map((item) => item.falseCost), 0.01);
  return (
    <section className="panel chart-panel wide-chart">
      <div className="chart-heading">
        <h3>Task Completion vs False Action Cost</h3>
        <p>Higher success is better; lower false action cost is better.</p>
      </div>
      <div className="tradeoff-plot" aria-label="Task completion and false action cost tradeoff">
        {c4TaskChartRows.map((item) => {
          const left = `${Math.min(96, Math.max(4, item.success * 100))}%`;
          const top = `${Math.min(94, Math.max(6, (item.falseCost / maxCost) * 88))}%`;
          return (
            <span className="tradeoff-point" key={item.method} style={{ left, top }}>
              <i />
              <strong>{item.method}</strong>
            </span>
          );
        })}
        <div className="axis-label x-axis">Task success</div>
        <div className="axis-label y-axis">False cost</div>
      </div>
    </section>
  );
}

function GuidePage() {
  return (
    <section className="results-page guide-page">
      <div className="results-header">
        <MousePointer2 size={22} />
        <div>
          <h2>Gesture Guide</h2>
          <p>Use slower, separated gestures. Return to Point or an open idle hand between commands.</p>
        </div>
      </div>
      <div className="guide-grid">
        {gestureGuideCards.map((item) => {
          const Icon = item.icon;
          return (
            <article className={`guide-card ${item.id}`} key={item.id}>
              <div className="gesture-visual">
                <Icon size={34} strokeWidth={1.8} />
                <span className="gesture-motion" />
              </div>
              <div>
                <span>{item.action}</span>
                <strong>{item.label}</strong>
                <p>{item.pose}</p>
              </div>
              <em>{item.cue}</em>
            </article>
          );
        })}
      </div>
      <section className="panel guide-flow">
        <div>
          <span>Recommended rhythm</span>
          <strong>Point → command → pause</strong>
          <p>Most mistakes come from holding Click too long or blending one gesture into the next.</p>
        </div>
        <div>
          <span>Live control</span>
          <strong>Use TARC Controller</strong>
          <p>Direct Control is left only for raw baseline checks; TARC is the intended demo mode.</p>
        </div>
      </section>
    </section>
  );
}

function ChartsPage() {
  return (
    <section className="results-page charts-page">
      <div className="results-header">
        <BarChart3 size={22} />
        <div>
          <h2>Methodology Charts</h2>
          <p>Recognition, action-safety and task-level AR results from the prepared IPN benchmark reports.</p>
        </div>
      </div>
      <div className="insight-strip">
        <article>
          <span>Main contribution</span>
          <strong>M3 TARC</strong>
          <p>Task-aware risk calibration keeps baseline-level completion while cutting false action cost.</p>
        </article>
        <article>
          <span>Best recognition</span>
          <strong>M2 Robust C6</strong>
          <p>Augmented TCN fusion lifts weak-class recognition and lowers false actions.</p>
        </article>
        <article>
          <span>Best safety</span>
          <strong>M3 Proposed</strong>
          <p>Lowest false-cost operating point among official methods.</p>
        </article>
      </div>
      <div className="chart-grid">
        <MetricBarChart
          title="Recognition Macro F1"
          subtitle="Classifier-level performance on public IPN test."
          values={recognitionChartRows}
          valueKey="macroF1"
        />
        <MetricBarChart
          title="Action Replay False Cost"
          subtitle="Weighted false AR action cost in replay evaluation."
          values={c4RiskChartRows}
          valueKey="falseCost"
          lowerIsBetter
        />
        <MetricBarChart
          title="Task-Level False Cost"
          subtitle="Weighted false action cost over AR scenario trials."
          values={c4TaskChartRows}
          valueKey="falseCost"
          lowerIsBetter
        />
        <MetricBarChart
          title="Task Success Rate"
          subtitle="Full task completion across 13 AR scenarios."
          values={c4TaskChartRows}
          valueKey="success"
        />
        <MetricBarChart
          title="Weak Scenario Success"
          subtitle="Lowest C4 task-aware task success rates."
          values={weakTaskRows.map((item) => ({ method: item.task, success: item.success }))}
          valueKey="success"
        />
        <MetricBarChart
          title="Weak Scenario Missed Cost"
          subtitle="Missed action cost explains remaining task failures."
          values={weakTaskRows.map((item) => ({ method: item.task, missedCost: item.missedCost }))}
          valueKey="missedCost"
          lowerIsBetter
        />
        <TradeoffChart />
      </div>
      <div className="results-table panel thesis-claim-table">
        <table>
          <thead>
            <tr>
              <th>Claim</th>
              <th>Evidence</th>
              <th>Interpretation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Recognition alone is not the thesis</td>
              <td>M1 macro F1 0.850; M2 macro F1 0.887</td>
              <td>The recognizer is improved, but live AR still needs command safety.</td>
            </tr>
            <tr>
              <td>Task-aware control reduces accidental commands</td>
              <td>False cost 0.110 to 0.025</td>
              <td>TARC treats classifier output as an action proposal, not an immediate command.</td>
            </tr>
            <tr>
              <td>Usability trade-off is explicit</td>
              <td>Task success 0.531 with precision 0.974</td>
              <td>The proposed method optimizes safety while preserving guided-task completion.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function ResultsPage() {
  return (
    <section className="results-page">
      <div className="results-header">
        <BarChart3 size={22} />
        <div>
          <h2>Experiment Results</h2>
          <p>Full IPN Hand target subset: 2405 train clips, 1033 test clips.</p>
        </div>
      </div>
      <div className="results-grid">
        {methods.map((item) => (
          <article key={item.id} className="result-card">
            <div>
              <h3>{item.label}</h3>
              <p>{item.role}</p>
            </div>
            <dl>
              <div>
                <dt>Accuracy</dt>
                <dd>{item.accuracy}</dd>
              </div>
              <div>
                <dt>Macro F1</dt>
                <dd>{item.macroF1}</dd>
              </div>
              <div>
                <dt>Latency</dt>
                <dd>{item.latency}</dd>
              </div>
              <div>
                <dt>Artifact</dt>
                <dd>{item.artifact}</dd>
              </div>
            </dl>
          </article>
        ))}
      </div>
      <div className="stage-grid">
        {researchStages.map((item) => (
          <article key={item.title} className="stage-card">
            <span>{item.title}</span>
            <strong>{item.value}</strong>
            <p>{item.detail}</p>
          </article>
        ))}
      </div>
      <section className="panel official-summary">
        <div>
          <span>Official compact comparison</span>
          <h3>Three methods, one clear research question</h3>
          <p>
            The live system now mirrors the thesis framing: compare direct baseline recognition, robust recognition,
            and the proposed task-aware AR controller that suppresses accidental actions during guided tasks.
          </p>
        </div>
        <dl>
          <div>
            <dt>M1 Success</dt>
            <dd>{c4TaskBenchmarkResults.directSuccess}</dd>
          </div>
          <div>
            <dt>M2 Success</dt>
            <dd>{c4TaskBenchmarkResults.c3c2Success}</dd>
          </div>
          <div>
            <dt>M3 Success</dt>
            <dd>{c4TaskBenchmarkResults.taskAwareSuccess}</dd>
          </div>
          <div>
            <dt>M3 Precision</dt>
            <dd>{c4TaskBenchmarkResults.taskAwarePrecision}</dd>
          </div>
          <div>
            <dt>M3 False Cost</dt>
            <dd>{c4TaskBenchmarkResults.taskAwareFalseCost}</dd>
          </div>
        </dl>
      </section>
      <div className="results-table panel task-results-table">
        <table>
          <thead>
            <tr>
              <th>Task Variant</th>
              <th>Task Success</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>Unintended</th>
              <th>False Cost</th>
            </tr>
          </thead>
          <tbody>
            {c4TaskRows.map((row) => (
              <tr key={row.method}>
                <td>{row.method}</td>
                <td>{row.success}</td>
                <td>{row.precision}</td>
                <td>{row.recall}</td>
                <td>{row.unintended}</td>
                <td>{row.falseCost}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="results-table panel">
        <table>
          <thead>
            <tr>
              <th>Method</th>
              <th>Role</th>
              <th>Recognition</th>
              <th>Interaction Layer</th>
              <th>Use in Demo</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>M1 Baseline Direct</td>
              <td>Control baseline</td>
              <td>Temporal TCN, macro F1 0.850</td>
              <td>Direct gesture-to-action mapping</td>
              <td>Direct mode only for raw comparison</td>
            </tr>
            <tr>
              <td>M2 Robust Direct</td>
              <td>Recognition upgrade</td>
              <td>C6 ensemble, macro F1 0.887</td>
              <td>Direct action mapping</td>
              <td>Available as Robust C6 + Direct</td>
            </tr>
            <tr>
              <td>M3 Proposed TARC</td>
              <td>Main thesis method</td>
              <td>Robust C6 proposals</td>
              <td>Task-aware thresholds, stability and cooldown</td>
              <td>Default live mode</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TaskRunnerControls({
  task,
  taskRun,
  live,
  source,
  detectionRate,
  onStart,
  onStop,
  onReset
}: {
  task: TaskDefinition;
  taskRun: TaskRunState;
  live: boolean;
  source: SourceId;
  detectionRate: string;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}) {
  const complete = taskRun.completed >= task.steps.length;
  const detectionValue = Number(detectionRate);
  const noHand = live && source === "webcam" && Number.isFinite(detectionValue) && detectionValue < 0.15;
  const status = complete ? "complete" : taskRun.active ? (noHand ? "no hand" : "running") : "ready";

  return (
    <section className="panel task-run-panel">
      <div className="panel-title">AR Task Run</div>
      <div className={`task-run-status ${complete ? "complete" : noHand ? "warn" : taskRun.active ? "active" : ""}`}>
        {complete ? <CheckCircle2 size={17} /> : noHand ? <AlertTriangle size={17} /> : <Timer size={17} />}
        <strong>{status}</strong>
        <span>{Math.round(taskRun.elapsedMs / 1000)}s</span>
      </div>
      <div className="task-run-actions">
        <button type="button" onClick={live ? onStop : onStart}>
          {live ? <WifiOff size={16} /> : <Play size={16} />}
          <span>{live ? "Stop Live" : "Start Task"}</span>
        </button>
        <button type="button" onClick={onReset}>
          <RotateCcw size={16} />
          <span>Reset Task</span>
        </button>
      </div>
      <div className="task-run-mini">
        <div>
          <span>Steps</span>
          <strong>
            {taskRun.completed}/{task.steps.length}
          </strong>
        </div>
        <div>
          <span>Misses</span>
          <strong>{taskRun.mistakes}</strong>
        </div>
      </div>
    </section>
  );
}

function TaskProgressOverlay({
  task,
  taskRun,
  live,
  detectionRate,
  policyContext
}: {
  task: TaskDefinition;
  taskRun: TaskRunState;
  live: boolean;
  detectionRate: string;
  policyContext: PolicyContext | null;
}) {
  const complete = taskRun.completed >= task.steps.length;
  const detectionValue = Number(detectionRate);
  const noHand = live && Number.isFinite(detectionValue) && detectionValue < 0.15;
  const policyStepIndex = typeof policyContext?.step_index === "number" ? policyContext.step_index : null;
  const policyStepCount = typeof policyContext?.step_count === "number" ? policyContext.step_count : task.steps.length;
  const expectedAction = policyContext?.expected_action || task.steps[taskRun.completed]?.action || "idle";
  const expectedLabel = policyContext?.expected_label || task.steps[taskRun.completed]?.gesture || "no_gesture";
  const expectedTitle = expectedAction === "idle" ? "Idle" : actionLabels[expectedAction];
  const currentGuide = gestureGuideCards.find((item) => item.id === expectedLabel);
  const CurrentGuideIcon = currentGuide?.icon ?? MousePointer2;

  return (
    <div className="task-progress-overlay">
      <div className="task-progress-header">
        <strong>{complete ? "Task Complete" : task.label}</strong>
        <span>{complete ? `${Math.round(taskRun.elapsedMs / 1000)}s` : expectedTitle}</span>
      </div>
      {!complete && currentGuide ? (
        <div className="current-gesture-cue">
          <CurrentGuideIcon size={28} strokeWidth={1.8} />
          <div>
            <span>Do now</span>
            <strong>{currentGuide.label}</strong>
            <p>{currentGuide.pose}</p>
          </div>
        </div>
      ) : null}
      {policyContext ? (
        <div className="policy-context-panel">
          <div>
            <span>Policy expects</span>
            <strong>{expectedTitle}</strong>
          </div>
          <div>
            <span>Gesture</span>
            <strong>{expectedLabel}</strong>
          </div>
          <div>
            <span>Policy step</span>
            <strong>
              {(policyStepIndex ?? 0) + 1}/{policyStepCount}
            </strong>
          </div>
          <div>
            <span>False events</span>
            <strong>{policyContext.false_events ?? 0}</strong>
          </div>
        </div>
      ) : null}
      <div className="task-step-list">
        {task.steps.map((step, index) => {
          const referenceIndex = policyStepIndex ?? taskRun.completed;
          const done = index < referenceIndex;
          const current = index === referenceIndex && !complete;
          return (
            <div key={step.id} className={done ? "done" : current ? "current" : ""}>
              {done ? <CheckCircle2 size={16} /> : <Circle size={16} />}
              <span>{step.label}</span>
              <strong>{step.gesture}</strong>
            </div>
          );
        })}
      </div>
      <div className={`task-live-signal ${noHand ? "warn" : live ? "on" : ""}`}>
        {noHand ? <AlertTriangle size={15} /> : <Radio size={15} />}
        <span>{noHand ? "no hand visible" : live ? "live recognition" : "standby"}</span>
      </div>
    </div>
  );
}

function App() {
  const [page, setPage] = useState<PageId>("demo");
  const [method, setMethod] = useState<MethodId>("c6_ensemble");
  const [source, setSource] = useState<SourceId>("webcam");
  const [interactionMode, setInteractionMode] = useState<InteractionMode>("c4_task_aware");
  const [task, setTask] = useState<TaskId>("object");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [gesture, setGesture] = useState<GestureId>("no_gesture");
  const [live, setLive] = useState(false);
  const [intervalMs, setIntervalMs] = useState(DEFAULT_INTERVAL_MS);
  const [cameraIndex, setCameraIndex] = useState(0);
  const [previewWidth, setPreviewWidth] = useState(DEFAULT_PREVIEW_WIDTH);
  const [jpegQuality, setJpegQuality] = useState(DEFAULT_JPEG_QUALITY);
  const threshold = 0.62;
  const stableFrames = 2;
  const [backendHealth, setBackendHealth] = useState<BackendHealth>("checking");
  const [backendStatus, setBackendStatus] = useState("stream stopped");
  const [backendConfidence, setBackendConfidence] = useState("--");
  const [sampleId, setSampleId] = useState("");
  const [targetLabel, setTargetLabel] = useState("");
  const [detectionRate, setDetectionRate] = useState("--");
  const [streamFps, setStreamFps] = useState("--");
  const [processingMs, setProcessingMs] = useState("--");
  const [sessionId, setSessionId] = useState("");
  const [previewImage, setPreviewImage] = useState("");
  const [landmarks, setLandmarks] = useState<number[][]>([]);
  const [cameraStats, setCameraStats] = useState<CameraStats | null>(null);
  const [pointerScreen, setPointerScreen] = useState<{ x: number; y: number } | null>(null);
  const [policyContext, setPolicyContext] = useState<PolicyContext | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [sceneState, setSceneState] = useState<SceneState>(initialSceneState);
  const [taskRun, setTaskRun] = useState<TaskRunState>(initialTaskRunState);

  const selectedTask = useMemo(() => liveTasks.find((item) => item.id === task) ?? liveTasks[0], [task]);
  const backendHealthText =
    backendHealth === "ready" ? "backend ready" : backendHealth === "checking" ? "checking backend" : "backend offline";
  const targetFps = Math.round(1000 / intervalMs);
  const sessionLabel = sessionId ? sessionId.split("_").slice(1, 3).join("_") : "--";
  const captureFps = Math.max(DEFAULT_TARGET_FPS, targetFps);
  const cameraResolution =
    cameraStats?.width && cameraStats?.height
      ? `${cameraStats.width}x${cameraStats.height}`
      : `${FAST_CAMERA_WIDTH}x${FAST_CAMERA_HEIGHT}`;
  const cameraFeedUrl =
    live && source === "webcam"
      ? `${BACKEND_HTTP_URL}/video_feed?camera=${cameraIndex}&camera_width=${FAST_CAMERA_WIDTH}&camera_height=${FAST_CAMERA_HEIGHT}` +
        `&preview_width=${previewWidth}&jpeg_quality=${jpegQuality}&mirror=true&fps=${targetFps}`
      : "";
  const cameraMessage =
    backendStatus === "backend unavailable"
      ? {
          title: "Backend is unavailable",
          body: "Start the Python backend, then press Start Camera again."
        }
      : backendStatus.startsWith("Cannot open camera")
        ? {
            title: "Camera cannot be opened",
            body: "Change the camera index or close another app that is using the camera."
          }
        : live
          ? {
              title: "Camera frames are not arriving",
              body: backendStatus
            }
          : {
              title: "Camera stream is stopped",
              body: "Start Camera activates the live AR layer."
            };

  useEffect(() => {
    let cancelled = false;

    const checkBackend = async () => {
      try {
        const response = await fetch(`${BACKEND_HTTP_URL}/api/health`, { cache: "no-store" });
        if (!cancelled) setBackendHealth(response.ok ? "ready" : "offline");
      } catch {
        if (!cancelled) setBackendHealth("offline");
      }
    };

    checkBackend();
    const interval = window.setInterval(checkBackend, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    setPreviewImage("");
    setLandmarks([]);
    setSampleId("");
    setTargetLabel("");
    setDetectionRate("--");
    setStreamFps("--");
    setProcessingMs("--");
    setSessionId("");
    setCameraStats(null);
    setPointerScreen(null);
    setPolicyContext(null);
  }, [source, cameraIndex]);

  useEffect(() => {
    if (!taskRun.active || !taskRun.startedAt) return;
    const interval = window.setInterval(() => {
      setTaskRun((current) =>
        current.active && current.startedAt ? { ...current, elapsedMs: Date.now() - current.startedAt } : current
      );
    }, 250);
    return () => window.clearInterval(interval);
  }, [taskRun.active, taskRun.startedAt]);

  useEffect(() => {
    if (!live) {
      wsRef.current?.close();
      wsRef.current = null;
      setBackendStatus((current) =>
        current === "backend unavailable" || current === "connection closed" || current.startsWith("Cannot open camera")
          ? current
          : "stream stopped"
      );
      setBackendConfidence("--");
      setPointerScreen(null);
      setPolicyContext(null);
      return;
    }

    setBackendStatus("connecting");
    const url =
      `${BACKEND_WS_URL}/ws/stream?method=${method}` +
      `&source=${source}&interval_ms=${intervalMs}&camera=${cameraIndex}` +
      `&interaction=${interactionMode}&threshold=${threshold}&stable_frames=${stableFrames}` +
      `&preview_width=${previewWidth}&jpeg_quality=${jpegQuality}&camera_width=${FAST_CAMERA_WIDTH}&camera_height=${FAST_CAMERA_HEIGHT}` +
      `&capture_fps=${captureFps}` +
      `&mirror=true&log=true&preview=false&task=${task}&max_log_mb=50`;
    const socket = new WebSocket(url);
    wsRef.current = socket;
    let closedByCleanup = false;

    socket.onopen = () => setBackendStatus(source);
    socket.onerror = () => setBackendStatus("backend unavailable");
    socket.onclose = () => {
      if (wsRef.current === socket && !closedByCleanup) {
        setBackendStatus((current) => (current === "backend unavailable" ? current : "connection closed"));
        setLive(false);
      }
    };
    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data) as StreamMessage;
      if (payload.type === "status") {
        setBackendStatus(payload.message ?? "working");
        return;
      }
      if (payload.type === "error") {
        setBackendStatus(payload.message ?? "error");
        setLive(false);
        return;
      }
      if (payload.gesture) {
        setGesture(payload.gesture);
      }
      setBackendConfidence(typeof payload.confidence === "number" ? payload.confidence.toFixed(2) : "--");
      setSampleId(payload.sample_id ?? "");
      setTargetLabel(payload.target_label ?? "");
      setDetectionRate(typeof payload.detection_rate === "number" ? payload.detection_rate.toFixed(2) : "--");
      setStreamFps(typeof payload.fps === "number" ? payload.fps.toFixed(1) : "--");
      setProcessingMs(typeof payload.processing_ms === "number" ? payload.processing_ms.toFixed(0) : "--");
      setSessionId(payload.session_id ?? "");
      setPreviewImage(payload.preview_image ?? "");
      setLandmarks(payload.landmarks ?? []);
      setCameraStats(payload.camera ?? null);
      setPointerScreen((current) => {
        if (!payload.pointer || source !== "webcam") return null;
        if (!current) return payload.pointer;
        return {
          x: current.x + (payload.pointer.x - current.x) * POINTER_SMOOTHING,
          y: current.y + (payload.pointer.y - current.y) * POINTER_SMOOTHING
        };
      });
      setPolicyContext(payload.policy_context ?? null);
      const nextAction = payload.event?.action ?? payload.action;
      if (nextAction && nextAction !== "idle") {
        setTaskRun((current) => advanceTaskRun(current, selectedTask, nextAction));
      }
      setSceneState((current) => {
        const rawPointerX = payload.pointer && source === "webcam" ? payload.pointer.x * 1.84 - 0.92 : null;
        const rawPointerY = payload.pointer && source === "webcam" ? (0.5 - payload.pointer.y) * 1.2 : null;
        const pointerX = rawPointerX !== null ? current.pointerX + (rawPointerX - current.pointerX) * POINTER_SMOOTHING : null;
        const pointerY = rawPointerY !== null ? current.pointerY + (rawPointerY - current.pointerY) * POINTER_SMOOTHING : null;
        const withPointer =
          pointerX !== null && pointerY !== null
            ? {
                ...current,
                pointerX,
                pointerY,
                targetIndex: Math.max(0, Math.min(4, Math.round(((pointerX + 0.92) / 1.84) * 4)))
              }
            : current;
        if (nextAction === "pointer_hover" && pointerX !== null && pointerY !== null) {
          return { ...withPointer, action: nextAction };
        }
        if (nextAction && nextAction !== "idle") {
          return applyAction(withPointer, nextAction, task);
        }
        return { ...withPointer, action: "idle" };
      });
    };

    return () => {
      closedByCleanup = true;
      socket.close();
    };
  }, [live, method, source, intervalMs, cameraIndex, interactionMode, threshold, stableFrames, previewWidth, jpegQuality, task, selectedTask, captureFps]);

  const dispatchGesture = (nextGesture: GestureId) => {
    const nextAction = gestureActions[nextGesture];
    setGesture(nextGesture);
    setBackendConfidence("--");
    setTaskRun((current) => advanceTaskRun(current, selectedTask, nextAction));
    setSceneState((current) => applyGesture(current, nextGesture, task));
  };

  const startTaskRun = () => {
    setSource("webcam");
    setSceneState(initialSceneState());
    setTaskRun({
      ...initialTaskRunState(),
      active: true,
      startedAt: Date.now()
    });
    if (backendHealth === "ready") {
      setLive(true);
      setBackendStatus("connecting");
    } else {
      setBackendStatus("backend unavailable");
    }
  };

  const resetTaskRun = () => {
    setTaskRun(initialTaskRunState());
    setSceneState(initialSceneState());
  };

  const stopTaskRun = () => {
    setLive(false);
    setBackendStatus("stream stopped");
    setTaskRun((current) => ({ ...current, active: false }));
  };

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-row">
          <ScanLine size={22} strokeWidth={1.9} />
          <div>
            <h1>Gesture AR</h1>
            <p>IPN Full Benchmark</p>
          </div>
        </div>

        <section className="panel page-tabs" aria-label="Page">
          <button type="button" className={page === "demo" ? "active" : ""} onClick={() => setPage("demo")}>
            <Sparkles size={16} />
            Live
          </button>
          <button type="button" className={page === "guide" ? "active" : ""} onClick={() => setPage("guide")}>
            <MousePointer2 size={16} />
            Guide
          </button>
          <button type="button" className={page === "results" ? "active" : ""} onClick={() => setPage("results")}>
            <BarChart3 size={16} />
            Tables
          </button>
          <button type="button" className={page === "charts" ? "active" : ""} onClick={() => setPage("charts")}>
            <BarChart3 size={16} />
            Charts
          </button>
        </section>

        <section className="panel session-panel">
          <div className="panel-title">Session</div>
          <label className="task-select-label">
            <span>AR task</span>
            <select
              value={task}
              onChange={(event) => {
                setTask(event.target.value as TaskId);
                setSceneState(initialSceneState());
                setTaskRun(initialTaskRunState());
              }}
            >
              {liveTasks.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <p className="task-description">{selectedTask.description}</p>
          <div className="session-controls">
            <div>
              <span>Recognizer</span>
              <div className="segmented" role="tablist" aria-label="Recognizer method">
                {methods.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={item.id === method ? "active" : ""}
                    onClick={() => setMethod(item.id)}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <span>Control</span>
              <div className="segmented" role="tablist" aria-label="Interaction mode">
                <button
                  type="button"
                  className={interactionMode === "c4_task_aware" ? "active" : ""}
                  onClick={() => setInteractionMode("c4_task_aware")}
                  title="Use the proposed task-aware risk-calibrated AR controller."
                >
                  <CheckCircle2 size={16} />
                  TARC
                </button>
                <button
                  type="button"
                  className={interactionMode === "direct" ? "active" : ""}
                  onClick={() => setInteractionMode("direct")}
                  title="Raw recognizer output without the task-aware controller."
                >
                  <MousePointer2 size={16} />
                  Direct
                </button>
              </div>
            </div>
          </div>
          <div className="run-summary compact-summary">
            <div>
              <span>Camera</span>
              <strong>{cameraResolution}</strong>
            </div>
            <div>
              <span>FPS</span>
              <strong>{targetFps}</strong>
            </div>
          </div>
        </section>

        <TaskRunnerControls
          task={selectedTask}
          taskRun={taskRun}
          live={live}
          source={source}
          detectionRate={detectionRate}
          onStart={startTaskRun}
          onStop={stopTaskRun}
          onReset={resetTaskRun}
        />

        <section className="panel live-status-panel">
          <div className="panel-title">Telemetry</div>
          <div className="live-status-grid">
            <div className={backendHealth === "ready" ? "status-pill on" : backendHealth === "checking" ? "status-pill" : "status-pill warn"}>
              <span>{backendHealthText}</span>
            </div>
            <div className={live ? "status-pill on" : "status-pill"}>
              <span>{live ? backendStatus : "camera stopped"}</span>
            </div>
            <div>
              <span>FPS</span>
              <strong>{streamFps}</strong>
            </div>
            <div>
              <span>Proc</span>
              <strong>{processingMs === "--" ? "--" : `${processingMs} ms`}</strong>
            </div>
            <div>
              <span>Detect</span>
              <strong>{detectionRate}</strong>
            </div>
            <div>
              <span>Gesture</span>
              <strong>{gesture}</strong>
            </div>
          </div>
        </section>

        <section className="panel advanced-panel">
          <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced((value) => !value)}>
            <Settings2 size={17} />
            <span>{showAdvanced ? "Hide Advanced" : "Advanced Controls"}</span>
          </button>
        </section>

        {showAdvanced ? (
          <>
            <section className="panel">
              <div className="panel-title">Camera Settings</div>
              <div className="stream-settings">
                <label>
                  <span>Target FPS</span>
                  <input
                    type="number"
                    min={12}
                    max={30}
                    value={targetFps}
                    onChange={(event) => {
                      const nextFps = Math.max(12, Math.min(30, Number(event.target.value) || DEFAULT_TARGET_FPS));
                      setIntervalMs(Math.round(1000 / nextFps));
                    }}
                  />
                </label>
                <label>
                  <span>Camera</span>
                  <input
                    type="number"
                    min={0}
                    max={8}
                    value={cameraIndex}
                    onChange={(event) => setCameraIndex(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>Preview px</span>
                  <input
                    type="number"
                    min={640}
                    max={1280}
                    step={80}
                    value={previewWidth}
                    onChange={(event) => setPreviewWidth(Number(event.target.value))}
                  />
                </label>
                <label>
                  <span>JPEG</span>
                  <input
                    type="number"
                    min={60}
                    max={92}
                    value={jpegQuality}
                    onChange={(event) => setJpegQuality(Number(event.target.value))}
                  />
                </label>
              </div>
            </section>

            <section className="panel">
              <div className="panel-title">Gesture Test Pad</div>
              <div className="gesture-grid">
                {gestureControls.map((item) => {
                  const Icon = item.icon;
                  return (
                    <button
                      type="button"
                      key={item.id}
                      title={item.label}
                      aria-label={item.label}
                      onClick={() => dispatchGesture(item.id)}
                    >
                      <Icon size={18} strokeWidth={1.9} />
                      <span>{item.label}</span>
                    </button>
                  );
                })}
              </div>
            </section>
          </>
        ) : null}
      </aside>

      {page === "charts" ? (
        <ChartsPage />
      ) : page === "guide" ? (
        <GuidePage />
      ) : page === "results" ? (
        <ResultsPage />
      ) : (
        <section className="work-surface">
          <SceneCanvas state={sceneState} task={task} />
          {previewImage || cameraFeedUrl ? (
            <img className="ar-camera-feed" src={previewImage || cameraFeedUrl} alt="Live camera AR background" />
          ) : null}
          {source === "webcam" && !previewImage && !cameraFeedUrl ? (
            <div className="camera-waiting">
              <Video size={28} />
              <strong>{cameraMessage.title}</strong>
              <span>{cameraMessage.body}</span>
            </div>
          ) : null}
          {(previewImage || cameraFeedUrl) &&
            landmarks.map((point, index) => (
              <span
                key={`${point[0]}-${point[1]}-${index}`}
                className="ar-landmark-dot"
                style={{ left: `${point[0] * 100}%`, top: `${point[1] * 100}%` }}
              />
            ))}
          {(previewImage || cameraFeedUrl) && pointerScreen ? (
            <span className="ar-pointer-reticle" style={{ left: `${pointerScreen.x * 100}%`, top: `${pointerScreen.y * 100}%` }}>
              <Crosshair size={28} strokeWidth={1.8} />
            </span>
          ) : null}
          <div className="task-badge">
            <strong>{selectedTask.label}</strong>
            <span>{selectedTask.description}</span>
          </div>
          <TaskProgressOverlay
            task={selectedTask}
            taskRun={taskRun}
            live={live}
            detectionRate={detectionRate}
            policyContext={policyContext}
          />
          <div className="telemetry">
            <div>
              <span>Gesture</span>
              <strong>{gesture}</strong>
            </div>
            <div>
              <span>Action</span>
              <strong>{sceneState.action}</strong>
            </div>
            <div>
              <span>Scale</span>
              <strong>{sceneState.scale.toFixed(2)}</strong>
            </div>
            <div>
              <span>Hits</span>
              <strong>{sceneState.hits}</strong>
            </div>
            <div>
              <span>Conf</span>
              <strong>{backendConfidence}</strong>
            </div>
            <div>
              <span>Detect</span>
              <strong>{detectionRate}</strong>
            </div>
            <div>
              <span>FPS</span>
              <strong>{streamFps}</strong>
            </div>
            <div>
              <span>Proc</span>
              <strong>{processingMs === "--" ? "--" : `${processingMs} ms`}</strong>
            </div>
            <div>
              <span>Input</span>
              <strong>{source === "webcam" ? `cam ${cameraIndex}` : targetLabel || "dataset"}</strong>
            </div>
            <div>
              <span>Log</span>
              <strong>{sessionLabel}</strong>
            </div>
            <div>
              <span>Policy</span>
              <strong>{policyContext?.expected_action || interactionModeLabels[interactionMode]}</strong>
            </div>
            <div className={sceneState.selected ? "selected-indicator on" : "selected-indicator"}>
              <Sparkles size={16} />
              <strong>{sceneState.selected ? "Selected" : "Ready"}</strong>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
