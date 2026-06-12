export type AgentRunStatus = "completed" | "waiting_for_human" | "failed";

export interface AgentRunResponse {
  status: AgentRunStatus;
  answer: string;
  citations: string[];
  tool_calls: Array<Record<string, unknown>>;
  needs_approval: boolean;
  approval_id: string | null;
  error: string | null;
}

export interface BackendHealth {
  status: string;
  protein?: Record<string, unknown>;
  agent?: Record<string, unknown>;
  environment?: string;
  provider?: string;
}

export interface RuntimeConfig {
  apiUrl: string | null;
  wsUrl: string | null;
}

export interface WorkflowEvent {
  id: string;
  event: string;
  receivedAt: string;
  payload: Record<string, unknown>;
}

export interface StatusScreen {
  slug: string;
  event: string;
  label: string;
  phase: string;
  tone: "neutral" | "active" | "success" | "warning" | "danger";
  description: string;
  nextAction: string;
}

export const STATUS_SCREENS: StatusScreen[] = [
  {
    slug: "accepted",
    event: "accepted",
    label: "Accepted",
    phase: "Socket",
    tone: "success",
    description: "The unified FastAPI WebSocket accepted the workflow request.",
    nextAction: "Wait for clarification or public research to start.",
  },
  {
    slug: "clarification-started",
    event: "clarification_started",
    label: "Clarifying",
    phase: "Agent",
    tone: "active",
    description: "The agent is checking whether the biological request is specific enough.",
    nextAction: "Review the clarification result before allowing the workflow to continue.",
  },
  {
    slug: "clarification-completed",
    event: "clarification_completed",
    label: "Clarified",
    phase: "Agent",
    tone: "success",
    description: "The agent produced a refined protein search query or marked the request as clear.",
    nextAction: "If the backend asks for input, approve, revise, or cancel from Human Review.",
  },
  {
    slug: "waiting-for-user",
    event: "waiting_for_user",
    label: "Human Review",
    phase: "HITL",
    tone: "warning",
    description: "The backend is paused and waiting for approve, revise, or cancel.",
    nextAction: "Use the dashboard review controls to send the next WebSocket action.",
  },
  {
    slug: "public-research-started",
    event: "public_research_started",
    label: "Researching",
    phase: "Evidence",
    tone: "active",
    description: "The workflow is using Exa for public background evidence only.",
    nextAction: "Watch for completion, failure, or skipped research before sequence fetch.",
  },
  {
    slug: "public-research-completed",
    event: "public_research_completed",
    label: "Evidence Ready",
    phase: "Evidence",
    tone: "success",
    description: "Public research results were normalized and attached to the workflow.",
    nextAction: "Continue to sequence fetching and local semantic ranking.",
  },
  {
    slug: "public-research-failed",
    event: "public_research_failed",
    label: "Research Failed",
    phase: "Evidence",
    tone: "warning",
    description: "Public research failed, but the backend marks the workflow as continuing.",
    nextAction: "Treat the result as lower evidence quality and continue monitoring.",
  },
  {
    slug: "public-research-skipped",
    event: "public_research_skipped",
    label: "Research Skipped",
    phase: "Evidence",
    tone: "neutral",
    description: "The workflow was configured to skip public research.",
    nextAction: "Continue with database fetch and local semantic ranking.",
  },
  {
    slug: "fetch-started",
    event: "fetch_started",
    label: "Fetching",
    phase: "Database",
    tone: "active",
    description: "The backend is fetching protein records from NCBI, ENA, or auto source selection.",
    nextAction: "Check source, query, and limit before semantic ranking begins.",
  },
  {
    slug: "fetch-completed",
    event: "fetch_completed",
    label: "Fetched",
    phase: "Database",
    tone: "success",
    description: "Protein records were fetched and are ready for ranking.",
    nextAction: "Inspect record count and move to semantic search.",
  },
  {
    slug: "semantic-search-started",
    event: "semantic_search_started",
    label: "Ranking",
    phase: "Semantic",
    tone: "active",
    description: "Local semantic search is ranking fetched protein records.",
    nextAction: "Wait for selected record and match list.",
  },
  {
    slug: "semantic-search-completed",
    event: "semantic_search_completed",
    label: "Ranked",
    phase: "Semantic",
    tone: "success",
    description: "The backend selected the best matching protein record and returned ranked matches.",
    nextAction: "Inspect selected accession before accepting the generated span prompt.",
  },
  {
    slug: "span-selected",
    event: "span_selected",
    label: "Span Selected",
    phase: "Masking",
    tone: "success",
    description: "The backend selected the masked protein span for completion.",
    nextAction: "Review start, end, mask length, and flanking context.",
  },
  {
    slug: "completed",
    event: "completed",
    label: "Completed",
    phase: "Output",
    tone: "success",
    description: "The workflow produced a span completion instruction and input payload.",
    nextAction: "Use the generated instruction/input pair in the next MDNAC stage.",
  },
  {
    slug: "cancelled",
    event: "cancelled",
    label: "Cancelled",
    phase: "HITL",
    tone: "neutral",
    description: "The workflow was cancelled by a human action.",
    nextAction: "Start a new request with tighter input if needed.",
  },
  {
    slug: "error",
    event: "error",
    label: "Error",
    phase: "Runtime",
    tone: "danger",
    description: "The backend sent an error event.",
    nextAction: "Read the detail payload, adjust the request, and rerun.",
  },
  {
    slug: "agent-started",
    event: "agent_started",
    label: "Agent Started",
    phase: "REST Agent",
    tone: "active",
    description: "The REST agent run request was submitted to /agent/run.",
    nextAction: "Wait for completion, human approval, or failure.",
  },
  {
    slug: "agent-waiting-for-human",
    event: "agent_waiting_for_human",
    label: "Agent Review",
    phase: "REST Agent",
    tone: "warning",
    description: "The REST agent returned a draft that requires approval.",
    nextAction: "Approve or reject the pending approval id.",
  },
  {
    slug: "agent-completed",
    event: "agent_completed",
    label: "Agent Completed",
    phase: "REST Agent",
    tone: "success",
    description: "The REST agent completed and returned an answer.",
    nextAction: "Review citations and tool calls before using the answer.",
  },
  {
    slug: "agent-failed",
    event: "agent_failed",
    label: "Agent Failed",
    phase: "REST Agent",
    tone: "danger",
    description: "The REST agent failed or the request could not be proxied.",
    nextAction: "Inspect the error payload and backend readiness.",
  },
];

export function statusSlug(event: string) {
  return event.replaceAll("_", "-");
}

export function statusBySlug(slug: string) {
  return STATUS_SCREENS.find((screen) => screen.slug === slug);
}

export function statusByEvent(event: string) {
  return STATUS_SCREENS.find((screen) => screen.event === event);
}

export function makeEvent(event: string, payload: Record<string, unknown> = {}): WorkflowEvent {
  return {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    event,
    receivedAt: new Date().toISOString(),
    payload,
  };
}

export function buildBrowserWsUrl(configuredUrl: string | null | undefined) {
  if (configuredUrl) {
    return configuredUrl;
  }

  if (typeof window === "undefined") {
    return "ws://127.0.0.1:8000/protein-span-completion/ws";
  }

  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.hostname}:8000/protein-span-completion/ws`;
}
