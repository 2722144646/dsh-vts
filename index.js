/**
 * DSH Host plugin: forwards assistant text and turn state to the local Python
 * avatar bridge. The bridge owns TTS, audio clock, VTS parameter injection and
 * the NDI self-view; this plugin is deliberately thin.
 */

const DEFAULT_BRIDGE_URL = 'ws://127.0.0.1:8765';

export const inject = ['tools'];

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
      } else if (message.type === 'snapshot.result') {
        const pending = this.pending.get(message.request_id);
        if (pending) {
          this.pending.delete(message.request_id);
          clearTimeout(pending.timer);
          pending.resolve(message.metrics ?? null);
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
      ctx.tools.register({
        name: 'vts_avatar_look',
        description:
          'Capture the current VTube Studio NDI self-view and return a JPEG path plus motion metrics. ' +
          'Use the read_image tool on the returned path when you need to actually see your avatar face, ' +
          'mouth, eyes, head or body pose.',
        parameters: { type: 'object', properties: {}, additionalProperties: false },
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
        async execute() {
          try {
            const metrics = await client.request('snapshot', {}, 6000);
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
              enum: ['happy', 'sad', 'angry', 'surprised', 'thinking', 'shy', 'wronged', 'calm', 'neutral'],
              description: 'Expression preset.',
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
