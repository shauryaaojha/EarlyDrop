/* EarlyDrop FX: runs inside the Streamlit page (injected by app/fx.py).
 * Ported from Documents/Animations/components:
 *   Background Animations/10  -> interactive dot grid (global background)
 *   Background Animations/1   -> WebGL light rays (home hero)
 *   Mouse Effects/13          -> lerped canvas cursor (ring + dot)
 *   Hero Animations/20        -> cinematic split-bar loader (Web Animations API, no GSAP)
 * Plus: scramble titles, count-up numbers, 3D tilt + glare cards, magnetic buttons,
 *       pointer spotlight and page-enter transitions.
 */
;(function () {
	if (window.EDFX) return
	const C = { cyan: '#e3c35a', violet: '#ecebe4', dot: '#1e2f28', ink: '#ecebe4' }   // chalk yellow → chalk white
	const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
	const coarse = window.matchMedia('(pointer: coarse)').matches
	const CAPTURE = location.search.includes('noloader')   // screenshot / PDF capture: no entrance motion
	const layers = {}
	let lastPage = null

	const make = (tag, css, parent = document.body) => {
		const e = document.createElement(tag)
		if (css) e.style.cssText = css
		parent.appendChild(e)
		return e
	}
	const hexRgb = h => {
		const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(h)
		return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [255, 255, 255]
	}

	/* ---------------------------------------------------------------- dot grid */
	function dotGrid() {
		const root = make('div', 'position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.9')
		root.id = 'ed-dotgrid'
		const canvas = make('canvas', 'display:block;width:100%;height:100%', root)
		const ctx = canvas.getContext('2d')
		const cfg = { size: 2.4, gap: 22, prox: 150, speedTrigger: 100, shockR: 260, shockS: 5, maxSpeed: 5000, resistance: 750, ret: 1.5 }
		const base = hexRgb(C.dot), act = hexRgb(C.cyan), act2 = hexRgb(C.violet)
		const dots = []
		const ptr = { x: -9999, y: -9999, lastT: 0, lastX: 0, lastY: 0 }
		let w = 1, h = 1, raf = 0, last = performance.now()
		const build = () => {
			const dpr = Math.max(1, window.devicePixelRatio || 1)
			w = window.innerWidth; h = window.innerHeight
			canvas.width = Math.floor(w * dpr); canvas.height = Math.floor(h * dpr)
			ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
			const cell = cfg.size + cfg.gap
			const cols = Math.floor((w + cfg.gap) / cell), rows = Math.floor((h + cfg.gap) / cell)
			const sx = (w - (cell * cols - cfg.gap)) / 2 + cfg.size / 2, sy = (h - (cell * rows - cfg.gap)) / 2 + cfg.size / 2
			dots.length = 0
			for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++)
				dots.push({ cx: sx + x * cell, cy: sy + y * cell, xo: 0, yo: 0, vx: 0, vy: 0 })
		}
		const tick = now => {
			raf = requestAnimationFrame(tick)
			const dtMs = Math.min(32, now - last); last = now
			const dt = dtMs / 1000
			ctx.clearRect(0, 0, w, h)
			const om = 8 / cfg.ret, k = om * om, c = 2 * om, drag = Math.exp(-dtMs / cfg.resistance)
			const p2 = cfg.prox * cfg.prox, r = cfg.size / 2
			for (let i = 0; i < dots.length; i++) {
				const d = dots[i]
				if (d.xo || d.yo || d.vx || d.vy) {
					d.vx += (-k * d.xo - c * d.vx) * dt; d.vy += (-k * d.yo - c * d.vy) * dt
					d.vx *= drag; d.vy *= drag
					d.xo += d.vx * dt; d.yo += d.vy * dt
					if (Math.abs(d.xo) + Math.abs(d.yo) + Math.abs(d.vx) + Math.abs(d.vy) < 0.01) d.xo = d.yo = d.vx = d.vy = 0
				}
				const dx = d.cx - ptr.x, dy = d.cy - ptr.y, ds = dx * dx + dy * dy
				let rad = r
				if (ds <= p2) {
					const t = 1 - Math.sqrt(ds) / cfg.prox
					const mix = d.cy / h
					const tr = act[0] * (1 - mix) + act2[0] * mix, tg = act[1] * (1 - mix) + act2[1] * mix, tb = act[2] * (1 - mix) + act2[2] * mix
					ctx.fillStyle = `rgb(${Math.round(base[0] + (tr - base[0]) * t)},${Math.round(base[1] + (tg - base[1]) * t)},${Math.round(base[2] + (tb - base[2]) * t)})`
					rad = r * (1 + t * 0.9)
				} else ctx.fillStyle = C.dot
				ctx.beginPath(); ctx.arc(d.cx + d.xo, d.cy + d.yo, rad, 0, 6.2832); ctx.fill()
			}
		}
		const onMove = e => {
			const now = performance.now(), dt = ptr.lastT ? now - ptr.lastT : 16
			let vx = ((e.clientX - ptr.lastX) / dt) * 1000, vy = ((e.clientY - ptr.lastY) / dt) * 1000
			let sp = Math.hypot(vx, vy)
			if (sp > cfg.maxSpeed) { vx *= cfg.maxSpeed / sp; vy *= cfg.maxSpeed / sp; sp = cfg.maxSpeed }
			ptr.lastT = now; ptr.lastX = e.clientX; ptr.lastY = e.clientY; ptr.x = e.clientX; ptr.y = e.clientY
			if (sp <= cfg.speedTrigger) return
			for (const d of dots) {
				const dist = Math.hypot(d.cx - ptr.x, d.cy - ptr.y)
				if (dist < cfg.prox) {
					const f = 1 - dist / cfg.prox
					d.vx += (d.cx - ptr.x + vx * 0.01) * f * 12; d.vy += (d.cy - ptr.y + vy * 0.01) * f * 12
				}
			}
		}
		const onClick = e => {
			for (const d of dots) {
				const dist = Math.hypot(d.cx - e.clientX, d.cy - e.clientY)
				if (dist < cfg.shockR) {
					const f = 1 - dist / cfg.shockR
					d.vx += (d.cx - e.clientX) * cfg.shockS * f * 18; d.vy += (d.cy - e.clientY) * cfg.shockS * f * 18
				}
			}
		}
		build()
		window.addEventListener('resize', build)
		window.addEventListener('mousemove', onMove, { passive: true })
		window.addEventListener('click', onClick)
		raf = requestAnimationFrame(tick)
		return () => {
			cancelAnimationFrame(raf)
			window.removeEventListener('resize', build); window.removeEventListener('mousemove', onMove); window.removeEventListener('click', onClick)
			root.remove()
		}
	}

	/* ---------------------------------------------------------------- light rays */
	const RAYS_FRAG = `precision highp float;
uniform float iTime; uniform vec2 iResolution, rayPos, rayDir, mousePos; uniform vec3 raysColor;
uniform float raysSpeed, lightSpread, rayLength, fadeDistance, mouseInfluence;
float rayStrength(vec2 src, vec2 dir, vec2 coord, float a, float b, float speed){
  vec2 s2c = coord - src; float cosA = dot(normalize(s2c), dir);
  float spread = pow(max(cosA, 0.0), 1.0 / max(lightSpread, 0.001));
  float dist = length(s2c); float maxD = iResolution.x * rayLength;
  float lenF = clamp((maxD - dist) / maxD, 0.0, 1.0);
  float fadeF = clamp((iResolution.x * fadeDistance - dist) / (iResolution.x * fadeDistance), 0.5, 1.0);
  float base = clamp((0.45 + 0.15 * sin(cosA * a + iTime * speed)) + (0.3 + 0.2 * cos(-cosA * b + iTime * speed)), 0.0, 1.0);
  return base * lenF * fadeF * spread;
}
void main(){
  vec2 coord = vec2(gl_FragCoord.x, iResolution.y - gl_FragCoord.y);
  vec2 dir = normalize(mix(rayDir, normalize(mousePos * iResolution.xy - rayPos), mouseInfluence));
  vec4 c = vec4(1.0) * rayStrength(rayPos, dir, coord, 36.2214, 21.11349, 1.5 * raysSpeed) * 0.5
         + vec4(1.0) * rayStrength(rayPos, dir, coord, 22.3991, 18.0234, 1.1 * raysSpeed) * 0.4;
  float br = 1.0 - coord.y / iResolution.y;
  c.x *= 0.1 + br * 0.8; c.y *= 0.3 + br * 0.6; c.z *= 0.5 + br * 0.5;
  c.rgb *= raysColor; gl_FragColor = c;
}`
	function lightRays() {
		const root = make('div', 'position:fixed;left:0;top:0;width:100vw;height:92vh;z-index:0;pointer-events:none;mix-blend-mode:screen;opacity:0;transition:opacity 1.2s ease')
		root.id = 'ed-rays'
		const canvas = make('canvas', 'width:100%;height:100%;display:block', root)
		const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false })
		if (!gl) { root.remove(); return () => {} }
		const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); return x }
		const prog = gl.createProgram()
		gl.attachShader(prog, sh(gl.VERTEX_SHADER, 'attribute vec2 position;void main(){gl_Position=vec4(position,0.,1.);}'))
		gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, RAYS_FRAG))
		gl.linkProgram(prog); gl.useProgram(prog)
		gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA)
		const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf)
		gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW)
		const loc = gl.getAttribLocation(prog, 'position'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0)
		const U = n => gl.getUniformLocation(prog, n)
		const col = hexRgb('#f3ecd2').map(v => v / 255)
		gl.uniform3fv(U('raysColor'), col); gl.uniform1f(U('raysSpeed'), 1.0); gl.uniform1f(U('lightSpread'), 0.55)
		gl.uniform1f(U('rayLength'), 2.2); gl.uniform1f(U('fadeDistance'), 1.0); gl.uniform1f(U('mouseInfluence'), 0.12)
		const mouse = { x: 0.5, y: 0.5 }, sm = { x: 0.5, y: 0.5 }
		let W = 1, H = 1, raf = 0
		const place = () => {
			const dpr = Math.min(window.devicePixelRatio || 1, 2)
			W = canvas.width = Math.floor(root.clientWidth * dpr); H = canvas.height = Math.floor(root.clientHeight * dpr)
			gl.viewport(0, 0, W, H)
			gl.uniform2f(U('iResolution'), W, H); gl.uniform2f(U('rayPos'), 0.5 * W, -0.2 * H); gl.uniform2f(U('rayDir'), 0, 1)
		}
		const onMove = e => { mouse.x = e.clientX / root.clientWidth; mouse.y = e.clientY / root.clientHeight }
		const loop = t => {
			sm.x = sm.x * 0.92 + mouse.x * 0.08; sm.y = sm.y * 0.92 + mouse.y * 0.08
			gl.uniform1f(U('iTime'), t * 0.001); gl.uniform2f(U('mousePos'), sm.x, sm.y)
			gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES, 0, 3)
			raf = requestAnimationFrame(loop)
		}
		place(); raf = requestAnimationFrame(loop)
		requestAnimationFrame(() => (root.style.opacity = '0.3'))
		window.addEventListener('resize', place); window.addEventListener('mousemove', onMove, { passive: true })
		return () => {
			cancelAnimationFrame(raf); window.removeEventListener('resize', place); window.removeEventListener('mousemove', onMove)
			try { gl.getExtension('WEBGL_lose_context')?.loseContext() } catch (e) {}
			root.remove()
		}
	}

	/* ---------------------------------------------------------------- cursor */
	function cursor() {
		if (coarse) return () => {}
		const canvas = make('canvas', 'position:fixed;inset:0;z-index:2147483646;pointer-events:none;mix-blend-mode:difference')
		canvas.id = 'ed-cursor'
		const ctx = canvas.getContext('2d')
		const m = { x: innerWidth / 2, y: innerHeight / 2 }, ring = { x: m.x, y: m.y, r: 14, tr: 14 }, dot = { x: m.x, y: m.y }
		let raf = 0, down = false
		const size = () => { const d = devicePixelRatio || 1; canvas.width = innerWidth * d; canvas.height = innerHeight * d; canvas.style.width = innerWidth + 'px'; canvas.style.height = innerHeight + 'px'; ctx.setTransform(d, 0, 0, d, 0, 0) }
		const HOVER = 'a,button,[role="tab"],[role="option"],[role="slider"],input,textarea,select,label,summary,[data-baseweb="select"],.ed-kpi,.ed-card,.ed-chip,[data-testid="stSidebarNavLink"]'
		const onMove = e => { m.x = e.clientX; m.y = e.clientY; ring.tr = e.target.closest && e.target.closest(HOVER) ? 34 : 14 }
		const lerp = (a, b, n) => a + (b - a) * n
		const render = () => {
			ring.x = lerp(ring.x, m.x, 0.18); ring.y = lerp(ring.y, m.y, 0.18); ring.r = lerp(ring.r, down ? ring.tr * 0.7 : ring.tr, 0.2)
			dot.x = lerp(dot.x, m.x, 0.6); dot.y = lerp(dot.y, m.y, 0.6)
			ctx.clearRect(0, 0, innerWidth, innerHeight)
			ctx.beginPath(); ctx.arc(ring.x, ring.y, ring.r, 0, 6.2832); ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.4; ctx.stroke()
			ctx.beginPath(); ctx.arc(dot.x, dot.y, 3, 0, 6.2832); ctx.fillStyle = '#ffffff'; ctx.fill()
			raf = requestAnimationFrame(render)
		}
		const d1 = () => (down = true), d0 = () => (down = false)
		size(); raf = requestAnimationFrame(render)
		window.addEventListener('resize', size); window.addEventListener('mousemove', onMove, { passive: true })
		window.addEventListener('mousedown', d1); window.addEventListener('mouseup', d0)
		return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', size); window.removeEventListener('mousemove', onMove); window.removeEventListener('mousedown', d1); window.removeEventListener('mouseup', d0); canvas.remove() }
	}

	/* ---------------------------------------------------------------- loader */
	function loader() {
		if (location.search.includes('noloader')) return
		try { if (sessionStorage.getItem('ed-loaded')) return } catch (e) {}
		try { sessionStorage.setItem('ed-loaded', '1') } catch (e) {}
		const words = Array(9).fill('<span>EARLYDROP</span>').join('')
		const wrap = make('div', 'position:fixed;inset:0;z-index:2147483645;background:#0f1a16;overflow:hidden;cursor:pointer')
		wrap.id = 'ed-loader'
		wrap.innerHTML = `
		<style>
		#ed-loader .clip{position:absolute;left:0;width:100vw;overflow:hidden;background:#15231e}
		#ed-loader .top{top:0;height:33.3vh}#ed-loader .bot{bottom:0;height:33.3vh}
		#ed-loader .mid{position:absolute;top:33.3vh;height:33.4vh;width:100vw;overflow:hidden}
		#ed-loader .mq{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:220vw;display:flex;justify-content:space-between;
		  font:700 clamp(2rem,9vw,10rem)/1 'Space Grotesk',sans-serif;letter-spacing:-.04em;white-space:nowrap}
		#ed-loader .mq span{margin-inline:28px}
		#ed-loader .top .mq,#ed-loader .bot .mq{color:transparent;-webkit-text-stroke:1.5px rgba(227,195,90,.55)}
		#ed-loader .mid .mq{color:#ecebe4}
		#ed-loader .meta{position:absolute;left:28px;bottom:22px;font:500 12px 'IBM Plex Mono',monospace;color:#8fa197;letter-spacing:.14em}
		#ed-loader .bar{position:absolute;left:0;bottom:0;height:2px;width:0;background:#e3c35a}
		</style>
		<div class="clip top"><div class="mq">${words}</div></div>
		<div class="mid"><div class="mq">${words}</div></div>
		<div class="clip bot"><div class="mq">${words}</div></div>
		<div class="meta">21CSC305P · MACHINE LEARNING · LOADING MODELS</div><div class="bar"></div>`
		const q = s => wrap.querySelector(s)
		const ease = 'cubic-bezier(.76,0,.24,1)'
		const T = reduce ? 0.2 : 1
		q('.top').animate([{ height: '50vh' }, { height: '33.3vh' }], { duration: 1100 * T, easing: ease, fill: 'both' })
		q('.bot').animate([{ height: '50vh' }, { height: '33.3vh' }], { duration: 1100 * T, easing: ease, fill: 'both' })
		q('.top .mq').animate([{ left: '120%' }, { left: '45%' }], { duration: 2300 * T, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'both' })
		q('.bot .mq').animate([{ left: '-20%' }, { left: '55%' }], { duration: 2300 * T, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'both' })
		q('.mid .mq').animate([{ left: '-30%', opacity: 0.4 }, { left: '50%', opacity: 1 }], { duration: 2200 * T, easing: 'cubic-bezier(.65,0,.35,1)', fill: 'both' })
		q('.bar').animate([{ width: '0' }, { width: '100%' }], { duration: 2200 * T, easing: 'cubic-bezier(.3,0,.2,1)', fill: 'both' })
		const out = () => {
			if (wrap._out) return; wrap._out = true
			q('.top').animate([{ clipPath: 'inset(0 0 0 0)' }, { clipPath: 'inset(0 0 100% 0)' }], { duration: 900, easing: ease, fill: 'both' })
			q('.bot').animate([{ clipPath: 'inset(0 0 0 0)' }, { clipPath: 'inset(100% 0 0 0)' }], { duration: 900, easing: ease, fill: 'both' })
			q('.mid').animate([{ opacity: 1, transform: 'scale(1)' }, { opacity: 0, transform: 'scale(1.15)' }], { duration: 700, easing: ease, fill: 'both' })
			wrap.animate([{ backgroundColor: '#0f1a16' }, { backgroundColor: 'rgba(15,26,22,0)' }], { duration: 900, delay: 250, fill: 'both' }).onfinish = () => wrap.remove()
			enterPage(true)
		}
		wrap.addEventListener('click', out)
		setTimeout(out, 2300 * T)
		setTimeout(() => wrap.remove(), 3500)   // hard cap: never leave the loader on screen
	}

	/* ---------------------------------------------------------------- UI micro-interactions */
	const GLYPHS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&$@'
	function scramble(el) {
		if (el.dataset.edScr || reduce || CAPTURE) return
		el.dataset.edScr = '1'
		const target = el
		const text = target.textContent
		if (!text || text.length > 24) return
		target.style.minWidth = target.offsetWidth + 'px'
		const dur = 900, t0 = performance.now()
		const step = now => {
			const p = Math.min(1, (now - t0) / dur)
			const n = Math.floor(p * text.length)
			let s = text.slice(0, n)
			for (let i = n; i < text.length; i++) s += text[i] === ' ' ? ' ' : GLYPHS[(Math.random() * GLYPHS.length) | 0]
			target.textContent = s
			if (p < 1) requestAnimationFrame(step); else target.textContent = text
		}
		requestAnimationFrame(step)
	}
	function countUp(el) {
		if (el.dataset.edDone) return
		el.dataset.edDone = '1'
		const to = parseFloat(el.dataset.to), dec = parseInt(el.dataset.dec || '0'), pre = el.dataset.pre || '', suf = el.dataset.suf || ''
		if (!isFinite(to) || reduce || CAPTURE) return
		const dur = 1400, t0 = performance.now()
		const step = now => {
			const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 4)
			el.textContent = pre + (to * e).toFixed(dec) + suf
			if (p < 1) requestAnimationFrame(step)
		}
		requestAnimationFrame(step)
	}
	function reveal(h) {
		if (h.dataset.edRev || reduce || CAPTURE) return
		h.dataset.edRev = '1'
		h.animate([{ clipPath: 'inset(0 0 100% 0)', transform: 'translateY(40px)' }, { clipPath: 'inset(0 0 -20% 0)', transform: 'none' }],
			{ duration: 900, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'backwards' })
	}
	function scan() {
		document.querySelectorAll('.ed-hero-title').forEach(scramble)
		document.querySelectorAll('[data-testid="stMainBlockContainer"] h1:not(.ed-hero-title)').forEach(reveal)
		document.querySelectorAll('.ed-count').forEach(countUp)
	}
	function enterPage(force) {
		const main = document.querySelector('[data-testid="stMainBlockContainer"]')
		if (!main || reduce || CAPTURE) return
		const kids = [...main.querySelectorAll(':scope > div > div > [data-testid="stElementContainer"], :scope > div > div > [data-testid="stHorizontalBlock"], :scope > div > div > [data-testid="stLayoutWrapper"]')].slice(0, 18)
		const items = kids.length ? kids : [main]
		items.forEach((k, i) => k.animate(
			[{ opacity: 0, transform: 'translateY(26px)', filter: 'blur(8px)' }, { opacity: 1, transform: 'none', filter: 'blur(0)' }],
			{ duration: 700, delay: (force ? 150 : 40) + i * 55, easing: 'cubic-bezier(.2,.8,.2,1)', fill: 'backwards' }))
	}
	function interactions() {
		const root = document.documentElement
		document.addEventListener('mousemove', e => {
			root.style.setProperty('--ed-mx', e.clientX + 'px'); root.style.setProperty('--ed-my', e.clientY + 'px')
			const card = e.target.closest && e.target.closest('.ed-kpi, .ed-card, .ed-tilt')
			if (card && !reduce) {
				const r = card.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height
				card.style.transform = `perspective(900px) rotateX(${(0.5 - y) * 9}deg) rotateY(${(x - 0.5) * 11}deg) translateZ(0)`
				card.style.setProperty('--gx', x * 100 + '%'); card.style.setProperty('--gy', y * 100 + '%')
				card._tilted = true
				if (window._edLastCard && window._edLastCard !== card) window._edLastCard.style.transform = ''
				window._edLastCard = card
			} else if (window._edLastCard) { window._edLastCard.style.transform = ''; window._edLastCard = null }
			const btn = e.target.closest && e.target.closest('.stButton button, .stDownloadButton button, .stFormSubmitButton button')
			if (btn && !reduce) {
				const r = btn.getBoundingClientRect()
				btn.style.transform = `translate(${(e.clientX - r.left - r.width / 2) * 0.18}px, ${(e.clientY - r.top - r.height / 2) * 0.3}px)`
				window._edBtn = btn
			} else if (window._edBtn) { window._edBtn.style.transform = ''; window._edBtn = null }
		}, { passive: true })
		new MutationObserver(() => { clearTimeout(window._edScanT); window._edScanT = setTimeout(scan, 60) })
			.observe(document.body, { childList: true, subtree: true })
	}

	/* ---------------------------------------------------------------- router */
	function on(name, factory) { if (!layers[name]) layers[name] = factory() }
	function off(name) { if (layers[name]) { layers[name](); delete layers[name] } }

	window.EDFX = {
		route(cfg) {
			document.documentElement.classList.toggle('ed-fx-on', !!cfg.fx)
			if (!cfg.fx) { Object.keys(layers).forEach(off); return }
			if (!window._edInteractions) { window._edInteractions = true; interactions() }
			on('dots', dotGrid)
			on('cursor', cursor)
			if (cfg.page === 'home') on('rays', lightRays); else off('rays')
			if (!window._edLoaderRan) { window._edLoaderRan = true; if (cfg.loader) loader() }
			if (cfg.page !== lastPage) { lastPage = cfg.page; setTimeout(() => enterPage(false), 30) }
			setTimeout(scan, 80)
		},
	}
})()
