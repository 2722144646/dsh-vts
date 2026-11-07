/**
 * DSH Host plugin: forwards assistant text and turn state to the local Python
 * avatar bridge. The bridge owns TTS, audio clock, VTS parameter injection and
 * the NDI self-view; this plugin is deliberately thin.
 */

const DEFAULT_BRIDGE_URL = 'ws://127.0.0.1:8765';

// Soft, optional guidance for the emotion self-report channel. Deliberately not
// a rule: not expressing is also an expression, so the model may stay silent.
const EMOTION_GUIDANCE = [
  '## 情绪自述（可选，不强制）',
  '',
  '你可以用 vts_avatar_feel 描述此刻的情绪。labels 是权重字典，例如 {"happy": 0.6, "tired": 0.4}，',
  '可以带 intensity、confidence、note。',
  '',
  '- 不调用同样合法：沉默、克制、没有情绪本身就是一种表达，不需要为了“有反应”硬填一个词。',
  '- 可以混合、可以不确定、可以低置信度；不必讨好。',
  '- 不需要每句话都调用；觉得情绪有变化时写一次即可，尽量在回复开始前，让表情和话音同步。',
  '- note 只写进日志，不会被念出来。',
].join('\n');

export const inject = ['tools', 'systemPrompt'];

function sessionIdOf(agent) {
  return String(
    agent?.session?.id ??
      agent?.sessionId ??
      agent?.id ??
      'default',
  );
}

class BridgeClient {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.retry = null;
    this.retryDelay = 1000;
    this.stopped = false;
    this.connected = false;
    this.lastStatus = null;
    this.pending = new Map();
    this.nextRequest = 1;
    this.mood = null;
  }

  start() {
    this.stopped = false;
    this.connect();
  }

  connect() {
    if (this.stopped) return;
    try {
      this.ws = new WebSocket(this.url);
    } catch (error) {
      console.warn('[vts-avatar] cannot create WebSocket:', error);
      this.scheduleReconnect();
      return;
    }
    this.ws.addEventListener('open', () => {
      this.connected = true;
      this.retryDelay = 1000;
      console.info('[vts-avatar] bridge connected:', this.url);
      this.send({
        type: 'hello',
        client: 'dsh-vts-avatar',
        protocol: 1,
      });
    });
    this.ws.addEventListener('message', (event) => {
      let message;
      try {
        message = JSON.parse(String(event.data));
      } catch {
        return;
      }
      if (message.type === 'hello.ack' || message.type === 'ready') {
        this.lastStatus = message.status ?? message;
        console.info('[vts-avatar] bridge ready:', JSON.stringify(this.lastStatus));
        this.request('mood.get').catch(() => {});
      } else if (message.type === 'mood.result') {
        this.setMood(message.text);
        const pending = this.pending.get(message.request_id);
        if (pending) {
          this.pending.delete(message.request_id);
          clearTimeout(pending.timer);
          pending.resolve(message);
        }
      } else if (message.type === 'mood.update') {
        this.setMood(message.text);
      } else if (message.type === 'snapshot.result') {
        const pending = this.pending.get(message.request_id);
        if (pending) {
          this.pending.delete(message.request_id);
          clearTimeout(pending.timer);
          pending.resolve(message.metrics ?? null);
        }
      } else if (message.type === 'record.result') {
        const pending = this.pending.get(message.request_id);
        if (pending) {
          this.pending.delete(message.request_id);
          clearTimeout(pending.timer);
          pending.resolve(message.summary ?? null);
        }
      } else if (message.type === 'feel.result') {
        const pending = this.pending.get(message.request_id);
        if (pending) {
          this.pending.delete(message.request_id);
          clearTimeout(pending.timer);
          pending.resolve(message);
        }
      } else if (message.type === 'error') {
        console.warn('[vts-avatar] bridge error:', message.code, message.message ?? '');
      }
    });
    this.ws.addEventListener('close', () => {
      this.connected = false;
      this.ws = null;
      if (!this.stopped) this.scheduleReconnect();
    });
    this.ws.addEventListener('error', () => {
      // `close` follows; avoid noisy stack traces.
    });
  }

  scheduleReconnect() {
    if (this.stopped || this.retry) return;
    const delay = this.retryDelay;
    this.retryDelay = Math.min(10000, this.retryDelay * 1.7);
    this.retry = setTimeout(() => {
      this.retry = null;
      this.connect();
    }, delay);
    this.retry.unref?.();
  }

  send(message) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    try {
      this.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      console.warn('[vts-avatar] send failed:', error);
      return false;
    }
  }

  /** Cache the latest mood summary so the prompt context can be computed. */
  setMood(text) {
    this.mood = { text: String(text || ''), at: Date.now() };
  }

  request(type, payload = {}, timeoutMs = 5000) {
    const requestId = `req-${this.nextRequest++}`;
    return new Promise((resolve, reject) => {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        reject(new Error('bridge is not connected'));
        return;
      }
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`bridge request ${type} timed out`));
      }, timeoutMs);
      this.pending.set(requestId, { resolve, reject, timer });
      if (!this.send({ type, request_id: requestId, ...payload })) {
        clearTimeout(timer);
        this.pending.delete(requestId);
        reject(new Error('bridge send failed'));
      }
    });
  }

  stop() {
    this.stopped = true;
    this.connected = false;
    if (this.retry) {
      clearTimeout(this.retry);
      this.retry = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        // ignore
      }
      this.ws = null;
    }
  }
}

export function apply(ctx, config = {}) {
  const url =
    (typeof config.bridgeUrl === 'string' && config.bridgeUrl) ||
    process.env.DSH_VTS_BRIDGE_URL ||
    DEFAULT_BRIDGE_URL;
  const client = new BridgeClient(url);
  /** @type {Map<string, {id: string, turn: number, step: number}>} */
  const active = new Map();

  const startUtterance = (agent, frame) => {
    const sid = sessionIdOf(agent);
    const previous = active.get(sid);
    if (previous?.started) {
      client.send({ type: 'cancel', utterance_id: previous.id, reason: 'new_turn' });
    }
    const id = `${sid}:${String(frame.attemptId ?? Date.now())}`;
    active.set(sid, {
      id,
      turn: Number(frame.turn ?? 0),
      step: Number(frame.step ?? 0),
      sent: '',
      started: false,
    });
    return id;
  };

  const ensureStarted = (current) => {
    if (current.started) return;
    current.started = true;
    client.send({
      type: 'utterance.start',
      utterance_id: current.id,
      turn: current.turn,
      step: current.step,
    });
  };

  const endUtterance = (agent, reason = 'end') => {
    const sid = sessionIdOf(agent);
    const current = active.get(sid);
    if (!current) return;
    active.delete(sid);
    if (!current.started) return;
    client.send({
      type: 'utterance.end',
      utterance_id: current.id,
      reason,
    });
  };

  ctx.effect(() => {
    client.start();

    const disposers = [
      ctx.systemPrompt.context({
        name: 'plugin:dsh-vts-avatar:mood',
        order: 260,
        text: () => {
          const mood = client.mood;
          if (!mood || !mood.text) return '';
          // Stale summaries would pin an old feeling on the model forever.
          if (Date.now() - mood.at > 10 * 60 * 1000) return '';
          return mood.text;
        },
      }),
      ctx.systemPrompt.section({
        name: 'plugin:dsh-vts-avatar',
        order: 250,
        text: EMOTION_GUIDANCE,
      }),
      ctx.tools.register({
        name: 'vts_avatar_look',
        description:
          'Capture the current VTube Studio NDI self-view and return a JPEG path plus motion metrics. ' +
          'Use the read_image tool on the returned path when you need to actually see your avatar face, ' +
          'mouth, eyes, head or body pose. Set frames > 1 to capture a short motion sequence and return a contact sheet.',
        parameters: {
          type: 'object',
          properties: {
            frames: { type: 'integer', minimum: 1, maximum: 12, description: 'Number of frames to capture; default 1.' },
            interval_ms: { type: 'integer', minimum: 0, maximum: 1000, description: 'Delay between frames; default 180ms.' },
          },
          additionalProperties: false,
        },
        output: {
          schema: {
            type: 'object',
            additionalProperties: true,
            properties: {
              ok: { type: 'boolean' },
              path: { type: 'string' },
              width: { type: 'number' },
              height: { type: 'number' },
              brightness: { type: 'number' },
              motion: { type: 'number' },
              mean_motion: { type: 'number' },
              mouth_motion: { type: 'number' },
              mean_mouth_motion: { type: 'number' },
              head_motion: { type: 'number' },
              mean_head_motion: { type: 'number' },
              frames: { type: 'number' },
              sequence: { type: 'string' },
              source: { type: 'string' },
              error: { type: 'string' },
            },
          },
          render: (_args, value) => [
            {
              type: 'text',
              text: value?.ok
                ? `Avatar snapshot: ${value.path} (${value.width}x${value.height}, motion=${value.motion}). Use read_image on this path.`
                : `Avatar snapshot failed: ${value?.error ?? 'unknown error'}`,
            },
          ],
        },
        async execute(args) {
          try {
            const frames = Math.max(1, Math.min(12, Number(args?.frames ?? 1)));
            const intervalMs = Math.max(0, Math.min(1000, Number(args?.interval_ms ?? 180)));
            const metrics = await client.request('snapshot', { frames, interval_ms: intervalMs }, 12000);
            if (!metrics) {
              return { ok: false, error: 'NDI self-view is not available. Enable NDI output in VTS.' };
            }
            return { ok: true, ...metrics };
          } catch (error) {
            return { ok: false, error: String(error?.message ?? error) };
          }
        },
      }),

      ctx.tools.register({
        name: 'vts_avatar_say',
        description:
          'Make the VTube Studio avatar speak one short line now. Use this for a manual voice test; normal DSH replies are sent automatically.',
        parameters: {
          type: 'object',
          properties: {
            text: { type: 'string', description: 'Chinese text to speak.' },
          },
          required: ['text'],
          additionalProperties: false,
        },
        output: {
          schema: {
            type: 'object',
            additionalProperties: true,
            properties: {
              ok: { type: 'boolean' },
              utterance_id: { type: 'string' },
              error: { type: 'string' },
            },
          },
          render: (_args, value) => [
            {
              type: 'text',
              text: value?.ok ? `Avatar speech queued: ${value.utterance_id}` : `Avatar speech failed: ${value?.error}`,
            },
          ],
        },
        async execute(args) {
          const text = String(args?.text ?? '').trim();
          if (!text) return { ok: false, error: 'text is empty' };
          const utteranceId = `manual:${Date.now()}`;
          const started = client.send({ type: 'utterance.start', utterance_id: utteranceId });
          const delta = client.send({ type: 'text.delta', utterance_id: utteranceId, text });
          const ended = client.send({ type: 'utterance.end', utterance_id: utteranceId });
          if (!started || !delta || !ended) {
            return { ok: false, error: 'bridge is not connected' };
          }
          return { ok: true, utterance_id: utteranceId };
        },
      }),

      ctx.tools.register({
        name: 'vts_avatar_emote',
        description:
          'Set the avatar expression explicitly: happy, sad, angry, surprised, thinking, shy, calm, or neutral. ' +
          'Use this as a side channel when the reply emotion is known and should not wait for keyword detection.',
        parameters: {
          type: 'object',
          properties: {
            name: {
              type: 'string',
              description: 'Preset or learned expression name.',
            },
            intensity: {
              type: 'number',
              minimum: 0,
              maximum: 1,
              description: 'Expression intensity; 0.7 is a good default.',
            },
          },
          required: ['name'],
          additionalProperties: false,
        },
        output: {
          schema: {
            type: 'object',
            additionalProperties: true,
            properties: {
              ok: { type: 'boolean' },
              error: { type: 'string' },
            },
          },
          render: (_args, value) => [
            {
              type: 'text',
              text: value?.ok ? `Avatar emotion set: ${value.name}` : `Avatar emotion failed: ${value?.error}`,
            },
          ],
        },
        async execute(args) {
          const name = String(args?.name ?? 'neutral').trim().toLowerCase();
          const intensity = Number(args?.intensity ?? 0.7);
          const sent = client.send({ type: 'emotion', name, intensity });
          if (!sent) return { ok: false, error: 'bridge is not connected' };
          return { ok: true, name, intensity };
        },
      }),

      ctx.tools.register({
        name: 'vts_avatar_feel',
        description:
          'Describe how the avatar feels right now with weighted labels, e.g. {happy: 0.6, tired: 0.4}. ' +
          'This is a description, not a preset command: mixed labels blend into one face, and a low ' +
          'confidence value softens the expression instead of snapping to a full preset. ' +
          'Optionally pass cause/why so the feeling keeps a reason in memory. ' +
          'Use vts_avatar_emote only when an exact preset has to be forced.',
        parameters: {
          type: 'object',
          properties: {
            labels: {
              type: 'object',
              description: 'Label weights 0..1, e.g. {"happy":0.6,"tired":0.4}. Keys are emotion words.',
              additionalProperties: { type: 'number', minimum: 0, maximum: 1 },
            },
            intensity: {
              type: 'number',
              minimum: 0,
              maximum: 1,
              description: 'Overall intensity; 0.7 is a good default.',
            },
            confidence: {
              type: 'number',
              minimum: 0,
              maximum: 1,
              description: 'How sure you are; low values soften the display instead of hard-switching.',
            },
            note: {
              type: 'string',
              description: 'Short reason for the affect log only; never spoken.',
            },
            cause: {
              type: 'string',
              description:
                'What happened, in one short sentence (e.g. 他说“你根本不在乎”). Optional, '
                + 'but this is the reason kept in memory — without it the feeling has no cause later.',
            },
            why: {
              type: 'string',
              description: 'Why it mattered to you (e.g. 我觉得被否定). Optional.',
            },
            appraisal: {
              type: 'array',
              items: { type: 'string' },
              description: 'Optional appraisal words, e.g. ["被否定","委屈"].',
            },
          },
          required: ['labels'],
          additionalProperties: false,
        },
        output: {
          schema: { type: 'object', additionalProperties: true },
          render: (_args, value) => {
            if (value?.ok === false) {
              return [{ type: 'text', text: `Avatar feel failed: ${value?.error ?? 'unknown error'}` }];
            }
            const dominant = Array.isArray(value?.dominant)
              ? value.dominant.map(([name, weight]) => `${name} ${Number(weight).toFixed(2)}`).join(', ')
              : '';
            return [
              {
                type: 'text',
                text: `Avatar feel applied (${value?.mode ?? 'unknown'}): ${dominant}` +
                  ` intensity ${value?.intensity ?? '?'}`,
              },
            ];
          },
        },
        async execute(args) {
          const labels = args?.labels && typeof args.labels === 'object' ? args.labels : {};
          try {
            const result = await client.request('feel', {
              labels,
              intensity: Number(args?.intensity ?? 0.7),
              confidence: Number(args?.confidence ?? 0.8),
              note: String(args?.note ?? ''),
              cause: String(args?.cause ?? ''),
              why: String(args?.why ?? ''),
              appraisal: Array.isArray(args?.appraisal) ? args.appraisal.map(String) : [],
            });
            if (result?.mood_text) client.setMood(result.mood_text);
            return { ...result, ok: result?.ok !== false };
          } catch (error) {
            return { ok: false, error: String(error?.message ?? error) };
          }
        },
      }),

      ctx.tools.register({
        name: 'vts_avatar_record',
        description:
          'Record a user-performed expression in read-only mode. The bridge pauses all parameter writes, ' +
          'the user performs the expression in VTS, and the resulting parameter deltas are saved on the current model.',
        parameters: {
          type: 'object',
          properties: {
            name: { type: 'string', description: 'Name for this learned expression.' },
            seconds: { type: 'number', minimum: 1, maximum: 10, description: 'Recording duration; default 4.' },
          },
          required: ['name'],
          additionalProperties: false,
        },
        output: {
          schema: { type: 'object', additionalProperties: true },
          render: (_args, value) => [
            {
              type: 'text',
              text: value?.ok
                ? `Recorded ${value.name}; top inputs: ${(value.summary?.top_inputs ?? []).slice(0, 6).join(', ')}`
                : `Recording failed: ${value?.error}`,
            },
          ],
        },
        async execute(args) {
          const name = String(args?.name ?? 'learned').trim();
          const seconds = Number(args?.seconds ?? 4);
          try {
            const summary = await client.request(
              'record',
              { name, seconds },
              Math.max(8000, (Number.isFinite(seconds) ? seconds : 4) * 1000 + 6000),
            );
            if (!summary) return { ok: false, error: 'empty record result' };
            return { ok: true, name, summary };
          } catch (error) {
            return { ok: false, error: String(error?.message ?? error) };
          }
        },
      }),

      ctx.on('agent/assistant-stream', (payload) => {
        const agent = payload?.agent;
        const frame = payload?.frame;
        if (!frame) return;
        const sid = sessionIdOf(agent);
        if (frame.type === 'start') {
          startUtterance(agent, frame);
          return;
        }
        if (frame.type === 'chunk') {
          const chunk = frame.chunk;
          if (!chunk) return;
          if (chunk.type === 'text-delta' && typeof chunk.text === 'string' && chunk.text.length > 0) {
            let current = active.get(sid);
            if (!current) current = startUtterance(agent, frame);
            ensureStarted(current);
            current.sent = (current.sent ?? '') + chunk.text;
            client.send({
              type: 'text.delta',
              utterance_id: current.id,
              text: chunk.text,
            });
          }
          if (chunk.type === 'block-end' && chunk.block?.type === 'text' && typeof chunk.block.text === 'string') {
            const current = active.get(sid);
            const already = current?.sent ?? '';
            const full = chunk.block.text;
            const remainder = full.startsWith(already) ? full.slice(already.length) : already ? '' : full;
            if (remainder && current) {
              ensureStarted(current);
              current.sent = already + remainder;
              client.send({
                type: 'text.delta',
                utterance_id: current.id,
                text: remainder,
              });
            }
          }
          return;
        }
        if (frame.type === 'end') {
          endUtterance(agent, frame.outcome?.kind ?? 'end');
        }
      }),

      ctx.on('agent/turn-stopping', (payload) => {
        endUtterance(payload?.agent, 'turn-stopping');
      }),

      ctx.on('agent/status', (payload) => {
        const status = payload?.status;
        if (status === 'running') {
          // Do not switch the performance layer to "thinking" while a spoken
          // utterance is active; that would damp the mouth mid-sentence.
          if (![...active.values()].some((current) => current.started)) {
            client.send({ type: 'state', state: 'thinking' });
          }
        } else if (status === 'idle' && active.size === 0) {
          client.send({ type: 'state', state: 'idle' });
        }
      }),

      ctx.on('agent/error', (payload) => {
        const sid = sessionIdOf(payload?.agent);
        const current = active.get(sid);
        if (current) {
          active.delete(sid);
          if (current.started) {
            client.send({
              type: 'cancel',
              utterance_id: current.id,
              reason: 'agent_error',
            });
          }
        }
      }),
    ];

    return () => {
      for (const dispose of disposers) {
        try {
          dispose?.();
        } catch {
          // ignore
        }
      }
      for (const current of active.values()) {
        client.send({ type: 'cancel', utterance_id: current.id, reason: 'plugin_dispose' });
      }
      active.clear();
      client.stop();
    };
  });
}
