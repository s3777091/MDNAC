"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import { BarChart2, Image as ImageIcon, MoreHorizontal, Send } from "lucide-react";

import { buildBrowserWsUrl, type RuntimeConfig } from "@/lib/mdnac";
import { cn } from "@/lib/utils";

interface CreativeCardProps {
  placeholder?: string;
  tags?: string[];
}

type SocketStatus = "loading" | "connecting" | "connected" | "running" | "waiting" | "closed" | "error";

const statusColor: Record<SocketStatus, string> = {
  loading: "bg-white/35",
  connecting: "bg-sky-300",
  connected: "bg-emerald-400",
  running: "bg-amber-300",
  waiting: "bg-yellow-300",
  closed: "bg-white/25",
  error: "bg-rose-400",
};

const CreativeCard: React.FC<CreativeCardProps> = ({
  placeholder = "Type your creative idea here...\u2728",
  tags = ["Generate Image", "Analyze Data", "Explore More"],
}) => {
  const [value, setValue] = useState("");
  const [wsUrl, setWsUrl] = useState<string | null>(null);
  const [socketStatus, setSocketStatus] = useState<SocketStatus>("loading");
  const socketRef = useRef<WebSocket | null>(null);

  const sendPayload = useCallback((socket: WebSocket, userInput: string) => {
    socket.send(JSON.stringify({ user_input: userInput }));
    setSocketStatus("running");
  }, []);

  const connectSocket = useCallback(
    (userInput?: string) => {
      if (!wsUrl) return;

      const currentSocket = socketRef.current;
      if (currentSocket?.readyState === WebSocket.OPEN) {
        if (userInput) {
          sendPayload(currentSocket, userInput);
        } else {
          setSocketStatus("connected");
        }
        return;
      }

      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;
      setSocketStatus("connecting");

      socket.onopen = () => {
        if (userInput) {
          sendPayload(socket, userInput);
          return;
        }
        setSocketStatus("connected");
      };

      socket.onmessage = (message) => {
        const event = readEventName(message.data);

        if (event === "waiting_for_user") {
          setSocketStatus("waiting");
          return;
        }

        if (event === "completed") {
          setSocketStatus("connected");
          return;
        }

        if (event === "error") {
          setSocketStatus("error");
        }
      };

      socket.onerror = () => {
        setSocketStatus("error");
      };

      socket.onclose = () => {
        if (socketRef.current === socket) {
          socketRef.current = null;
        }
        setSocketStatus((current) => (current === "error" ? "error" : "closed"));
      };
    },
    [sendPayload, wsUrl],
  );

  useEffect(() => {
    let ignore = false;

    async function loadConfig() {
      try {
        const response = await fetch("/api/config", { cache: "no-store" });
        const config = (await response.json()) as RuntimeConfig;
        if (!ignore) {
          setWsUrl(buildBrowserWsUrl(config.wsUrl));
        }
      } catch {
        if (!ignore) {
          setWsUrl(buildBrowserWsUrl(null));
        }
      }
    }

    loadConfig();

    return () => {
      ignore = true;
    };
  }, []);

  useEffect(() => {
    if (!wsUrl) return;

    connectSocket();

    return () => {
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connectSocket, wsUrl]);

  const handleSubmit = () => {
    const trimmed = value.trim();
    if (
      !trimmed ||
      !wsUrl ||
      socketStatus === "loading" ||
      socketStatus === "connecting" ||
      socketStatus === "running"
    ) {
      return;
    }

    if (socketStatus === "waiting") {
      socketRef.current?.send(JSON.stringify({ action: "revise", user_input: trimmed }));
      setSocketStatus("running");
      setValue("");
      return;
    }

    connectSocket(trimmed);
    setValue("");
  };

  const submitDisabled =
    !value.trim() ||
    !wsUrl ||
    socketStatus === "loading" ||
    socketStatus === "connecting" ||
    socketStatus === "running";

  return (
    <div className="mx-auto flex w-full max-w-[54rem] flex-col items-stretch">
      <div className="relative flex w-full flex-col overflow-hidden rounded-2xl p-[1px] sm:p-[2px]">
        <div className="pointer-events-none absolute -left-2 -top-2 size-8 rounded-full bg-white/45 blur-lg sm:size-10" />

        <div className="relative flex w-full flex-col overflow-hidden rounded-xl border border-slate-800 bg-black/55 shadow-[0_18px_70px_rgba(0,0,0,0.45)] backdrop-blur-2xl">
          <span
            className={cn(
              "absolute right-3 top-3 size-2.5 rounded-full shadow-[0_0_14px_currentColor] sm:right-4 sm:top-4 sm:size-3",
              statusColor[socketStatus],
            )}
            title={`WebSocket: ${socketStatus}`}
          />

          <div className="relative flex">
            <textarea
              id="chat_bot"
              name="chat_bot"
              placeholder={placeholder}
              className="min-h-28 w-full resize-none rounded-xl bg-transparent p-4 pr-12 font-sans text-sm font-semibold leading-relaxed text-white outline-none transition-all placeholder:text-slate-300 sm:min-h-36 sm:p-5 sm:pr-16 sm:text-base lg:min-h-44"
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  handleSubmit();
                }
              }}
            />
          </div>

          <div className="flex items-end justify-between gap-4 p-3 sm:p-5">
            <div className="flex gap-3 sm:gap-4">
              <button
                aria-label="Image"
                className="flex cursor-pointer border-none bg-transparent text-white/20 transition-all duration-300 hover:-translate-y-1 hover:text-white"
                type="button"
              >
                <ImageIcon className="size-5 sm:size-6" />
              </button>
              <button
                aria-label="Analyze"
                className="flex cursor-pointer border-none bg-transparent text-white/20 transition-all duration-300 hover:-translate-y-1 hover:text-white"
                type="button"
              >
                <BarChart2 className="size-5 sm:size-6" />
              </button>
              <button
                aria-label="More"
                className="flex cursor-pointer border-none bg-transparent text-white/20 transition-all duration-300 hover:-translate-y-1 hover:text-white"
                type="button"
              >
                <MoreHorizontal className="size-5 sm:size-6" />
              </button>
            </div>

            <button
              aria-label="Submit"
              className="flex size-11 shrink-0 cursor-pointer rounded-lg border-none bg-gradient-to-t from-slate-800 via-slate-600 to-slate-800 p-1 shadow-inner outline-none transition-all duration-150 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50 sm:size-12"
              disabled={submitDisabled}
              onClick={handleSubmit}
              type="button"
            >
              <i className="flex size-full items-center justify-center rounded-lg bg-white/10 p-2 text-slate-400 backdrop-blur-sm">
                <Send
                  className="size-5 transition-all duration-300 hover:text-white hover:drop-shadow-[0_0_5px_#fff] sm:size-6"
                />
              </i>
            </button>
          </div>
        </div>

        <div className="flex w-full flex-wrap gap-2 px-1 py-3 text-xs text-white sm:gap-3 sm:py-4 sm:text-sm">
          {tags.map((tag) => (
            <button
              key={tag}
              className="min-h-8 cursor-pointer select-none rounded-lg border border-slate-800 bg-black px-3 py-1.5 transition-colors hover:border-slate-600"
              onClick={() => setValue(tag)}
              type="button"
            >
              {tag}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

function readEventName(data: unknown) {
  if (typeof data !== "string") return "";

  try {
    const parsed = JSON.parse(data) as unknown;
    if (parsed && typeof parsed === "object" && "event" in parsed) {
      return String((parsed as { event?: unknown }).event ?? "");
    }
  } catch {
    return "";
  }

  return "";
}

export default CreativeCard;
