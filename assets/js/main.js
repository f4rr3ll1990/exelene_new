/* Player head follows the cursor.
   Frames are a head turn extracted from the reference video (index 0 = looking
   to the viewer's right, NEUTRAL = facing the camera, last = looking left),
   stabilised and drawn over a static body layer on one canvas. */
(() => {
  const canvas = document.querySelector('.player');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const FRAMES = { count: 41, x: 282, y: 48, w: 891, h: 733, neutral: 18 };
  const BODY = { x: 247, y: 640, w: 994, h: 571 };
  // Head centre in canvas coordinates (used as the "eye" of the tracking).
  const HEAD = { x: 680, y: 330 };
  const EASE = 0.12;

  const path = (i) => `assets/img/player/f${String(i).padStart(2, '0')}.webp`;
  const frames = new Array(FRAMES.count);
  let body = null;

  const load = (src) => new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => (img.decode ? img.decode().catch(() => {}) : Promise.resolve()).then(() => resolve(img));
    img.onerror = reject;
    img.src = src;
  });

  let current = FRAMES.neutral;
  let target = FRAMES.neutral;
  let drawn = -1;

  const nearestLoaded = (i) => {
    for (let d = 0; d < FRAMES.count; d++) {
      if (frames[i - d]) return i - d;
      if (frames[i + d]) return i + d;
    }
    return -1;
  };

  const draw = (force) => {
    const i = nearestLoaded(Math.round(current));
    if (i < 0 || (i === drawn && !force)) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (body) ctx.drawImage(body, BODY.x, BODY.y, BODY.w, BODY.h);
    ctx.drawImage(frames[i], FRAMES.x, FRAMES.y, FRAMES.w, FRAMES.h);
    drawn = i;
  };

  // Neutral frame and body first, the rest of the turn afterwards.
  Promise.all([load(path(FRAMES.neutral)), load('assets/img/player/body.webp')]).then(([f, b]) => {
    frames[FRAMES.neutral] = f;
    body = b;
    draw(true);
    for (let i = 0; i < FRAMES.count; i++) {
      if (i === FRAMES.neutral) continue;
      load(path(i)).then((img) => { frames[i] = img; }).catch(() => {});
    }
  });

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const aim = (clientX) => {
    const r = canvas.getBoundingClientRect();
    const headX = r.left + (HEAD.x / canvas.width) * r.width;
    const vw = document.documentElement.clientWidth;
    if (clientX >= headX) {
      const t = Math.min(1, (clientX - headX) / Math.max(1, vw - headX));
      target = FRAMES.neutral - t * FRAMES.neutral;
    } else {
      const t = Math.min(1, (headX - clientX) / Math.max(1, headX));
      target = FRAMES.neutral + t * (FRAMES.count - 1 - FRAMES.neutral);
    }
  };

  if (!reduceMotion) {
    window.addEventListener('pointermove', (e) => aim(e.clientX), { passive: true });
    window.addEventListener('touchmove', (e) => e.touches[0] && aim(e.touches[0].clientX), { passive: true });
    document.documentElement.addEventListener('pointerleave', () => { target = FRAMES.neutral; });
    window.addEventListener('blur', () => { target = FRAMES.neutral; });
  }

  const tick = () => {
    current += (target - current) * EASE;
    if (Math.abs(target - current) < 0.01) current = target;
    draw(false);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  // Mobile menu toggle
  const burger = document.querySelector('.burger');
  if (burger) {
    burger.addEventListener('click', () => {
      const open = document.body.classList.toggle('menu-open');
      burger.setAttribute('aria-expanded', String(open));
    });
  }
})();
