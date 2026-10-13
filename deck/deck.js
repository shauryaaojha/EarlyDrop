/* EarlyDrop review deck engine. Every number on every slide comes from window.EARLYDROP (deck-data.js,
 * written by run_all.py). Charts are hand-built SVG so they share the board look and animate in. */
;(function () {
	const D = window.EARLYDROP
	if (!D) { document.body.insertAdjacentHTML('beforeend', '<p style="padding:24px;color:#ecebe4">deck-data.js is missing. Run <code>python run_all.py</code> first.</p>'); return }
	D.meta.total_time = Object.values(D.meta.timings_sec).reduce((a, b) => a + b, 0)

	const PRINT = /print/.test(location.search) || location.hash === '#print'
	const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches
	const ANIM = !PRINT && !REDUCE && !!window.gsap
	const G = window.gsap
	const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim()
	const COL = { ink: css('--ink'), ink2: css('--ink2'), muted: css('--muted'), muted2: css('--muted2'), line: css('--line'), line2: css('--line2'),
		panel: css('--panel'), bg: css('--bg'), hi: css('--hi'), coral: css('--coral'), ochre: css('--ochre'), sky: css('--sky'),
		d: [css('--d1'), css('--d2'), css('--d3'), css('--d4'), css('--d5')], s: [css('--s0'), css('--s1'), css('--s2')] }
	const STATE_TXT = [COL.coral, COL.ochre, COL.sky]
	const STATES = ['Struggling', 'Stable', 'Strong']
	const BANDS = ['Poor', 'Average', 'Good']

	/* ------------------------------------------------------------ data binding */
	const get = path => path.split('.').reduce((o, k) => (o == null ? o : o[k]), D)
	const fmt = (v, f) => {
		if (v == null || (typeof v === 'number' && !isFinite(v))) return '–'
		if (typeof v !== 'number') return String(v)
		if (!f) return Number.isInteger(v) ? String(v) : String(+v.toFixed(3))
		if (f[0] === 'f') return v.toFixed(+f.slice(1))
		if (f[0] === 'p') return (100 * v).toFixed(+f.slice(1)) + '%'
		if (f[0] === 's') return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(+f.slice(1))
		return String(v)
	}
	document.querySelectorAll('[data-b]').forEach(el => { el.textContent = fmt(get(el.dataset.b), el.dataset.f) })
	const nice = s => s.replace(/_/g, ' ')

	/* ------------------------------------------------------------ svg helpers */
	const NS = 'http://www.w3.org/2000/svg'
	function svgIn(el, w, h) {
		el.innerHTML = ''
		const s = document.createElementNS(NS, 'svg')
		s.setAttribute('viewBox', `0 0 ${w} ${h}`)
		s.setAttribute('preserveAspectRatio', 'xMidYMid meet')
		el.appendChild(s)
		return s
	}
	function add(parent, tag, attrs = {}, text) {
		const e = document.createElementNS(NS, tag)
		for (const k in attrs) e.setAttribute(k, attrs[k])
		if (text != null) e.textContent = text
		parent.appendChild(e)
		return e
	}
	const size = el => [Math.max(200, el.clientWidth), Math.max(140, el.clientHeight || 300)]
	const scale = (d0, d1, r0, r1) => v => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0)
	const T = (p, x, y, s, o = {}) => add(p, 'text', { x, y, fill: o.fill || COL.muted, 'font-size': o.size || 14, 'text-anchor': o.anchor || 'start',
		'font-family': o.font || 'IBM Plex Mono, monospace', 'font-weight': o.weight || 400, 'dominant-baseline': o.base || 'middle' }, s)

	// animation helpers (no-ops in print / reduced motion)
	const grow = (els, attr, to, o = {}) => { if (!ANIM) return; G.from(els, { attr: { [attr]: o.from ?? 0 }, duration: o.d || 1, ease: 'power3.out', stagger: o.stagger ?? 0.05, delay: o.delay || 0.15 }) }
	const draw = (paths, o = {}) => {
		if (!ANIM) return
		;[].concat(paths).forEach((p, i) => {
			const L = p.getTotalLength(); p.style.strokeDasharray = L; p.style.strokeDashoffset = L
			G.to(p, { strokeDashoffset: 0, duration: o.d || 1.4, ease: 'power2.inOut', delay: (o.delay || 0.2) + i * (o.stagger || 0.15),
				onComplete: () => { p.style.strokeDasharray = o.dash || ''; p.style.strokeDashoffset = '' } })
		})
	}
	const pop = (els, o = {}) => { if (!ANIM) return; G.from(els, { opacity: 0, scale: 0.2, transformOrigin: 'center', transformBox: 'fill-box', duration: 0.5, ease: 'back.out(2)', stagger: o.stagger ?? 0.03, delay: o.delay || 0.3 }) }
	const fade = (els, o = {}) => { if (!ANIM) return; G.from(els, { opacity: 0, duration: 0.6, stagger: o.stagger ?? 0.04, delay: o.delay || 0.5 }) }

	/* ------------------------------------------------------------ generic charts */
	function axes(s, x0, y0, x1, y1, ys, yticks, xs, xticks, o = {}) {
		const g = add(s, 'g')
		yticks.forEach(t => {
			add(g, 'line', { x1: x0, x2: x1, y1: ys(t), y2: ys(t), stroke: COL.line, 'stroke-width': 1 })
			T(g, x0 - 10, ys(t), o.yfmt ? o.yfmt(t) : t, { anchor: 'end', size: 13, fill: COL.muted2 })
		})
		xticks.forEach(t => T(g, xs(t), y1 + 22, o.xfmt ? o.xfmt(t) : t, { anchor: 'middle', size: 13, fill: COL.muted2 }))
		if (o.xlabel) T(g, (x0 + x1) / 2, y1 + 46, o.xlabel, { anchor: 'middle', size: 13, fill: COL.muted })
		return g
	}
	function lineChart(el, series, o = {}) {
		const [w, h] = size(el), s = svgIn(el, w, h)
		const L = o.left || 64, R = o.right || 24, Tp = o.top || 18, B = o.bottom || (o.xlabel ? 58 : 36)
		const allX = series.flatMap(d => d.xs), allY = series.flatMap(d => d.ys)
		const tf = o.log ? Math.log10 : v => v
		let ymin = o.ymin ?? Math.min(...allY), ymax = o.ymax ?? Math.max(...allY)
		if (!o.log && o.ymin == null) { const pad = (ymax - ymin) * 0.12 || 1; ymin -= pad; ymax += pad }
		const xs = scale(Math.min(...allX), Math.max(...allX), L, w - R), ysRaw = scale(tf(ymin), tf(ymax), h - B, Tp), ys = v => ysRaw(tf(v))
		const yt = o.yticks || nice5(ymin, ymax)
		axes(s, L, Tp, w - R, h - B, ys, yt, xs, o.xticks || [...new Set(allX)], { xlabel: o.xlabel, yfmt: o.yfmt, xfmt: o.xfmt })
		if (o.markX != null) {
			add(s, 'line', { x1: xs(o.markX), x2: xs(o.markX), y1: Tp, y2: h - B, stroke: COL.hi, 'stroke-dasharray': '5 6', 'stroke-width': 2 })
			if (o.markLabel) T(s, xs(o.markX) + 8, Tp + 10, o.markLabel, { fill: COL.hi, size: 13 })
		}
		const paths = [], dots = []
		series.forEach(d => {
			if (d.area) add(s, 'path', { d: `M${xs(d.xs[0])},${h - B} ` + d.xs.map((x, i) => `L${xs(x)},${ys(d.ys[i])}`).join(' ') + ` L${xs(d.xs.at(-1))},${h - B}Z`, fill: d.color, opacity: 0.16 })
			const p = add(s, 'path', { d: d.xs.map((x, i) => `${i ? 'L' : 'M'}${xs(x)},${ys(d.ys[i])}`).join(' '), fill: 'none', stroke: d.color, 'stroke-width': d.width || 3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': d.dash || '' })
			paths.push(p)
			if (d.markers !== false) d.xs.forEach((x, i) => dots.push(add(s, 'circle', { cx: xs(x), cy: ys(d.ys[i]), r: d.r || 5.5, fill: d.colors ? d.colors[i] : d.color, stroke: COL.bg, 'stroke-width': 2.5 })))
			if (d.label) T(s, xs(d.xs.at(-1)) + 10, ys(d.ys.at(-1)), d.label, { fill: d.labelColor || d.color, size: 14, weight: 500 })
		})
		draw(paths); pop(dots, { delay: 0.6 })
		return { s, xs, ys }
	}
	function nice5(a, b) {
		const span = b - a, step0 = span / 4, mag = 10 ** Math.floor(Math.log10(step0)), st = [1, 2, 2.5, 5, 10].map(m => m * mag).find(m => m >= step0)
		const out = []; for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toFixed(6)); return out
	}
	function hbars(el, rows, o = {}) {
		const [w, h0] = size(el), rh = o.rowH || Math.min(44, (h0 - 20) / rows.length), h = Math.max(h0, rows.length * rh + 20), s = svgIn(el, w, h)
		const LW = o.labelW ?? 300, VW = o.valueW ?? 80
		const min = o.min ?? Math.min(0, ...rows.map(r => r.value)), max = o.max ?? Math.max(...rows.map(r => r.value + (r.err || 0)))
		const xs = scale(min, max, LW, w - VW), x0 = xs(Math.max(min, 0))
		if (min < 0) add(s, 'line', { x1: x0, x2: x0, y1: 4, y2: rows.length * rh + 8, stroke: COL.line2 })
		const bars = []
		rows.forEach((r, i) => {
			const y = 10 + i * rh, bh = Math.min(o.barH || 22, rh * 0.62), cy = y + rh / 2
			T(s, LW - 14, cy, r.label, { anchor: 'end', size: o.labelSize || 15, fill: r.labelColor || COL.ink2, font: 'IBM Plex Sans, sans-serif' })
			const xa = r.value >= 0 ? x0 : xs(r.value), wv = Math.abs(xs(r.value) - x0)
			const b = add(s, 'rect', { x: xa, y: cy - bh / 2, width: Math.max(1, wv), height: bh, rx: 3, fill: r.color || COL.d[0] })
			bars.push({ b, xa, wv, neg: r.value < 0 })
			if (r.err) add(s, 'line', { x1: xs(r.value - r.err), x2: xs(r.value + r.err), y1: cy, y2: cy, stroke: COL.ink, 'stroke-width': 2, opacity: 0.8 })
			T(s, r.value >= 0 ? Math.max(xs(r.value + (r.err || 0)), x0) + 10 : x0 + 10, cy, r.text ?? fmt(r.value, o.f || 'f3'), { size: 14, fill: COL.ink, weight: 500 })
		})
		if (ANIM) bars.forEach(({ b, xa, wv, neg }, i) => G.from(b, { attr: { width: 0, x: neg ? xa + wv : xa }, duration: 1, ease: 'power3.out', delay: 0.2 + i * 0.05 }))
		return s
	}
	function vbars(el, rows, o = {}) {
		const [w, h] = size(el), s = svgIn(el, w, h)
		const B = 56, Tp = 34, max = o.max ?? Math.max(...rows.map(r => r.value)) * 1.12, min = o.min ?? 0
		const ys = scale(min, max, h - B, Tp), bw = Math.min(120, (w - 40) / rows.length * 0.56), gap = (w - 40) / rows.length
		const bars = []
		rows.forEach((r, i) => {
			const cx = 20 + gap * (i + 0.5), y = ys(r.value)
			const b = add(s, 'rect', { x: cx - bw / 2, y, width: bw, height: h - B - y, rx: 4, fill: r.color || COL.d[0] })
			bars.push([b, y])
			T(s, cx, y - 16, r.text ?? fmt(r.value, o.f || 'f3'), { anchor: 'middle', size: 18, fill: COL.ink, weight: 500 })
			r.label.split('\n').forEach((ln, k) => T(s, cx, h - B + 22 + k * 18, ln, { anchor: 'middle', size: 14, fill: COL.ink2, font: 'IBM Plex Sans, sans-serif' }))
		})
		add(s, 'line', { x1: 10, x2: w - 10, y1: h - B, y2: h - B, stroke: COL.line2 })
		if (ANIM) bars.forEach(([b, y], i) => G.from(b, { attr: { y: h - B, height: 0 }, duration: 1.1, ease: 'power3.out', delay: 0.25 + i * 0.12 }))
		return s
	}
	function table(el, head, rows, o = {}) {
		const num = o.num || head.map((_, i) => i > 0)
		el.innerHTML = `<table class="t"><thead><tr>${head.map((hh, i) => `<th class="${num[i] ? 'n' : ''}">${hh}</th>`).join('')}</tr></thead><tbody>` +
			rows.map((r, ri) => `<tr class="${o.best === ri ? 'best' : ''}">${r.map((c, i) => `<td class="${num[i] ? 'n' : ''}">${c}</td>`).join('')}</tr>`).join('') + '</tbody></table>'
		fade(el.querySelectorAll('tbody tr'), { delay: 0.35, stagger: 0.06 })
	}

	/* ------------------------------------------------------------ chart registry */
	const U1 = D.unit1, U2 = D.unit2, U3 = D.unit3, U4 = D.unit4
	const ex = Object.fromEntries(U4.examples.map(e => [e.kind, e]))
	const CHARTS = {
		timeline(el) {
			const e = ex.declining || U4.examples[0], n = e.sgpa.length
			const { s, xs, ys } = lineChart(el, [{ xs: e.sgpa.map((_, i) => i + 1), ys: e.sgpa, color: COL.ink2, colors: e.viterbi.map(v => COL.s[v]), r: 8 }],
				{ ymin: 4, ymax: 10, yticks: [5, 6.5, 8, 10], xticks: e.sgpa.map((_, i) => i + 1), xfmt: t => 'Sem ' + t, left: 56, right: 300 })
			const x = xs(n), y = ys(e.sgpa[n - 1]), xm = xs(n - 0.5)
			add(s, 'line', { x1: 56, x2: xs(n), y1: ys(6.5), y2: ys(6.5), stroke: COL.coral, 'stroke-dasharray': '4 6', opacity: 0.6 })
			T(s, 62, ys(6.5) - 14, 'risk line · SGPA 6.5', { fill: COL.coral, size: 13 })
			const g1 = add(s, 'g'), g2 = add(s, 'g')
			add(g1, 'line', { x1: xm, x2: xm, y1: 10, y2: ys(4) , stroke: COL.hi, 'stroke-width': 2, 'stroke-dasharray': '6 6' })
			T(g1, xm + 12, 22, 'EarlyDrop flags it here', { fill: COL.hi, size: 15, weight: 500 })
			T(g1, xm + 12, 42, 'mid-semester survey signals', { fill: COL.muted, size: 13 })
			add(g2, 'circle', { cx: x, cy: y, r: 18, fill: 'none', stroke: COL.coral, 'stroke-width': 2.5 })
			T(g2, x + 30, y - 8, 'Reactive system notices', { fill: COL.coral, size: 15, weight: 500 })
			T(g2, x + 30, y + 12, 'after the result is out', { fill: COL.muted, size: 13 })
			if (ANIM) { G.from(g1, { opacity: 0, x: -20, delay: 1.4, duration: 0.7 }); G.from(g2, { opacity: 0, x: 20, delay: 2, duration: 0.7 }) }
		},
		calibration(el) {
			const rows = [...D.real.calibration].sort((a, b) => b.real_mean - a.real_mean)
			const half = Math.ceil(rows.length / 2)
			el.innerHTML = '<div class="row" style="height:calc(100% - 28px);gap:18px"><div class="grow c1" style="height:100%"></div><div class="grow c2" style="height:100%"></div></div>'
			;[rows.slice(0, half), rows.slice(half)].forEach((part, k) => {
				const box = el.querySelector(k ? '.c2' : '.c1'), [w, h] = size(box), s = svgIn(box, w, h), rh = (h - 40) / part.length
				const LW = 250, xs = scale(1, 5, LW, w - 14)
				;[1, 2, 3, 4, 5].forEach(t => { add(s, 'line', { x1: xs(t), x2: xs(t), y1: 6, y2: h - 30, stroke: COL.line }); T(s, xs(t), h - 14, t, { anchor: 'middle', size: 12, fill: COL.muted2 }) })
				const dots = []
				part.forEach((r, i) => {
					const cy = 14 + i * rh + rh / 2
					T(s, LW - 10, cy, nice(r.item).replace('programme ', 'prog. ').replace('academic ', 'acad. '), { anchor: 'end', size: 12.5, fill: COL.ink2, font: 'IBM Plex Sans, sans-serif' })
					add(s, 'line', { x1: xs(r.real_mean), x2: xs(r.synthetic_high_mean), y1: cy, y2: cy, stroke: COL.line2, 'stroke-width': 3 })
					dots.push(add(s, 'circle', { cx: xs(r.real_mean), cy, r: 6.5, fill: COL.hi }))
					dots.push(add(s, 'circle', { cx: xs(r.synthetic_high_mean), cy, r: 6.5, fill: 'none', stroke: COL.d[1], 'stroke-width': 2.5 }))
				})
				pop(dots, { stagger: 0.015 })
			})
			el.insertAdjacentHTML('afterbegin', `<div class="small" style="text-align:right;margin:-6px 0 6px"><span class="dot" style="background:${COL.hi}"></span>real (15)&nbsp;&nbsp;<span class="dot" style="border:2.5px solid ${COL.d[1]};width:9px;height:9px"></span>synthetic, CGPA ≥ 8.2</div>`)
		},
		cleaning(el) {
			const log = D.cleaning, by = k => log.find(l => l.step.startsWith(k)) || { count: 0 }
			const oor = log.filter(l => l.step.startsWith('Out-of-range'))
			const raw = D.meta.n_synthetic + by('Duplicate').count
			const steps = [
				{ n: raw, l: 'raw form rows', s: 'Google Form export, 50 columns' },
				{ n: by('Duplicate').count, l: 'duplicate submissions', s: 'same answers, seconds apart', c: COL.coral },
				{ n: by('Percent').count, l: '"80.50%" strings', s: 'parsed to numbers' },
				{ n: by('Rows missing').count, l: 'rows from before', s: '9 questions were added mid-survey', c: COL.ochre },
				{ n: oor.reduce((a, l) => a + l.count, 0), l: 'impossible values', s: oor.flatMap(l => l.values || []).join(' · '), c: COL.coral },
				{ n: by('Median').count, l: 'cells imputed', s: 'median / median category' },
				{ n: D.unit1.n_rows, l: 'clean students', s: '49 modelling columns, 0 missing', c: COL.hi },
			]
			el.innerHTML = '<div class="grid4" style="grid-template-columns:repeat(7,1fr);gap:14px;align-items:stretch">' + steps.map((st, i) =>
				`<div class="panel stp" style="padding:18px 16px;display:flex;flex-direction:column;gap:8px;${i === 6 ? 'border-color:var(--hi)' : ''}"><div class="cap">${String(i + 1).padStart(2, '0')}</div>` +
				`<div class="stat"><div class="v num cu2" data-to="${st.n}" style="font-size:46px;color:${st.c || 'var(--ink)'}">${st.n}</div></div>` +
				`<div style="font:600 17px var(--display);color:var(--ink)">${st.l}</div><div class="small" style="font-size:14px;line-height:1.4">${st.s}</div></div>`).join('') + '</div>'
			if (ANIM) { G.from(el.querySelectorAll('.stp'), { y: 40, opacity: 0, stagger: 0.12, duration: 0.7, ease: 'power3.out', delay: 0.2 }); countUp(el.querySelectorAll('.cu2'), 0.4) }
		},
		generator(el) {
			const [w, h] = size(el), s = svgIn(el, w, h)
			const box = (x, y, bw, bh, title, sub, color) => {
				const g = add(s, 'g')
				add(g, 'rect', { x, y, width: bw, height: bh, rx: 14, fill: COL.panel, stroke: color || COL.line2, 'stroke-width': 1.5 })
				T(g, x + 22, y + 34, title, { fill: color || COL.ink, size: 21, font: 'Space Grotesk, sans-serif', weight: 700 })
				sub.forEach((ln, i) => T(g, x + 22, y + 66 + i * 24, ln, { fill: COL.ink2, size: 15, font: 'IBM Plex Sans, sans-serif' }))
				return g
			}
			const arrow = (x1, y1, x2, y2) => add(s, 'path', { d: `M${x1},${y1} C${(x1 + x2) / 2},${y1} ${(x1 + x2) / 2},${y2} ${x2},${y2}`, fill: 'none', stroke: COL.hi, 'stroke-width': 2.5, 'stroke-dasharray': '7 7', 'marker-end': 'url(#gm)' })
			const defs = add(s, 'defs'), mk = add(defs, 'marker', { id: 'gm', viewBox: '0 0 10 10', refX: 7, refY: 5, markerWidth: 7, markerHeight: 7, orient: 'auto' }); add(mk, 'path', { d: 'M0,0 L10,5 L0,10z', fill: COL.hi })
			const bw = (w - 3 * 70) / 4, bh = 190, y = 20
			const g = [
				box(0, y, bw, bh, 'Latent traits', ['ability · engagement', 'motivation · support', 'one set per student']),
				box(bw + 70, y, bw, bh, 'Academic state zₜ', ['Struggling / Stable / Strong', 'Markov chain, matrix A', 'evolves each semester'], COL.hi),
				box(2 * (bw + 70), y, bw, bh, 'Semester SGPA', ['emitted from zₜ', '+ ability offset + noise', 'bands: < 6.5 · < 8 · ≥ 8']),
				box(3 * (bw + 70), y, bw, bh, 'Survey answers', ['50 form columns', 'calibrated on real answers', 'label: end-sem SGPA < 6.5']),
			]
			const arrows = [arrow(bw, y + bh / 2, bw + 66, y + bh / 2), arrow(2 * bw + 70, y + bh / 2, 2 * bw + 136, y + bh / 2), arrow(3 * bw + 140, y + bh / 2, 3 * bw + 206, y + bh / 2)]
			const A = D.unit4.truth.A_true
			const ty = y + bh + 70
			T(s, 0, ty, 'Ground truth we keep, so Units 3–4 can be checked:', { fill: COL.muted, size: 16, font: 'IBM Plex Sans, sans-serif' })
			const chips = ['true transition matrix A', 'true hidden state per semester', 'the 4 latent traits', 'calibration targets from the real form']
			let cx = 0
			chips.forEach(c => { const tw = c.length * 9.2 + 30; add(s, 'rect', { x: cx, y: ty + 22, width: tw, height: 36, rx: 6, fill: COL.panel, stroke: COL.line2 }); T(s, cx + 15, ty + 40, c, { fill: COL.ink2, size: 14 }); cx += tw + 12 })
			T(s, 0, ty + 96, `true self-transitions: A₁₁ = ${A[0][0]} · A₂₂ = ${A[1][1]} · A₃₃ = ${A[2][2]}`, { fill: COL.hi, size: 15 })
			if (ANIM) { G.from(g, { opacity: 0, y: 30, stagger: 0.25, duration: 0.7, ease: 'power3.out', delay: 0.2 }); draw(arrows, { delay: 0.6, stagger: 0.25, d: 0.8, dash: '7 7' }) }
		},
		corr(el) {
			const rows = Object.entries(U1.corr_with_target).sort((a, b) => a[1] - b[1]).map(([k, v]) => ({ label: nice(k), value: v, color: v < 0 ? COL.d[1] : COL.d[2], text: (v > 0 ? '+' : '') + v.toFixed(2) }))
			hbars(el, rows, { min: -0.75, max: 0.5, labelW: 330, f: 'f2' })
		},
		covpairs(el) {
			table(el, ['pair', 'cov', 'r'], U1.covariance_pairs.map(p => [`${nice(p.x).replace('_percentage', '')} ↔ ${nice(p.y)}`, p.cov.toFixed(2), p.r.toFixed(2)]))
			el.querySelector('table').style.fontSize = '15px'
		},
		bayes(el) {
			const c = U1.bayes.chain
			vbars(el, c.map((r, i) => ({ label: r.stage.replace('Prior P(H)', 'prior P(H)').replace('+ ', '+ ').replace(' < ', '\n< '), value: r.p, color: [COL.line2, COL.d[0], COL.d[2]][i], text: (100 * r.p).toFixed(0) + '%' })), { max: 1.12 })
		},
		poly(el) {
			const P = U1.poly_fit
			lineChart(el, [{ xs: P.degrees, ys: P.train_rmse, color: COL.d[1], label: 'train' }, { xs: P.degrees, ys: P.test_rmse, color: COL.d[2], label: 'test' }],
				{ log: true, ymin: 0.01, ymax: 20, yticks: [0.01, 0.1, 1, 10], yfmt: t => t, xlabel: 'polynomial degree M', markX: P.best_degree, markLabel: `best M = ${P.best_degree}`, right: 70 })
		},
		regtable(el) {
			const rows = U2.regression.table, best = rows.reduce((b, r, i) => (r.r2 > rows[b].r2 ? i : b), 0)
			table(el, ['model', 'R²', 'MAE', 'RMSE'], rows.map(r => [r.model, r.r2.toFixed(3), r.mae.toFixed(3), r.rmse.toFixed(3)]), { best })
		},
		robust(el) {
			const R = U2.regression.robustness
			const [w, h] = size(el), s = svgIn(el, w, h), B = 60, Tp = 30, max = Math.max(...Object.values(R).flatMap(r => [r.clean_rmse, r.corrupted_rmse])) * 1.18
			const ys = scale(0, max, h - B, Tp), names = Object.keys(R), gw = w / names.length, bw = 70, bars = []
			names.forEach((n, i) => {
				;[['clean_rmse', COL.d[1], 'clean'], ['corrupted_rmse', COL.d[2], 'corrupted']].forEach(([k, c, lab], j) => {
					const x = gw * i + gw / 2 + (j ? 6 : -bw - 6), y = ys(R[n][k])
					bars.push([add(s, 'rect', { x, y, width: bw, height: h - B - y, rx: 4, fill: c }), y])
					T(s, x + bw / 2, y - 14, R[n][k].toFixed(3), { anchor: 'middle', size: 15, fill: COL.ink })
					T(s, x + bw / 2, h - B + 18, lab, { anchor: 'middle', size: 12, fill: COL.muted2 })
				})
				T(s, gw * i + gw / 2, h - B + 42, n, { anchor: 'middle', size: 18, fill: COL.ink, font: 'Space Grotesk, sans-serif', weight: 700 })
			})
			if (ANIM) bars.forEach(([b, y], i) => G.from(b, { attr: { y: h - B, height: 0 }, duration: 1, ease: 'power3.out', delay: 0.3 + i * 0.15 }))
		},
		clf(el) {
			const rows = [...U2.classification.cv5].sort((a, b) => b.roc_auc_mean - a.roc_auc_mean)
			hbars(el, rows.map((r, i) => ({ label: r.model, value: r.roc_auc_mean, err: r.roc_auc_std, color: i === 0 ? COL.d[0] : COL.line2, text: `${r.roc_auc_mean.toFixed(3)} ± ${r.roc_auc_std.toFixed(3)}` })), { min: 0.85, max: 0.98, labelW: 290, valueW: 150 })
		},
		thresholds(el) {
			const T_ = [...U2.classification.blr_thresholds].sort((a, b) => a.threshold - b.threshold)
			lineChart(el, [{ xs: T_.map(r => r.threshold), ys: T_.map(r => r.recall), color: COL.d[0], label: 'recall' }, { xs: T_.map(r => r.threshold), ys: T_.map(r => r.precision), color: COL.d[1], label: 'precision' }],
				{ ymin: 0.5, ymax: 1, yticks: [0.5, 0.75, 1], xlabel: 'decision threshold', right: 100 })
		},
		elbow(el) { const k = U3.k_selection; lineChart(el, [{ xs: k.map(r => r.k), ys: k.map(r => r.kmeans_inertia), color: COL.d[1] }], { markX: 3, xlabel: 'k', yfmt: t => Math.round(t) }) },
		silh(el) { const k = U3.k_selection; lineChart(el, [{ xs: k.map(r => r.k), ys: k.map(r => r.kmeans_silhouette), color: COL.d[0], label: 'K-Means' }, { xs: k.map(r => r.k), ys: k.map(r => r.ward_silhouette), color: COL.d[3], label: 'Ward' }], { markX: 3, xlabel: 'k', right: 90 }) },
		bic(el) { const g = U3.gmm_selection; lineChart(el, [{ xs: g.map(r => r.k), ys: g.map(r => r.bic_full), color: COL.d[2] }], { markX: 3, markLabel: 'minimum', xlabel: 'k', yfmt: t => (t / 1000).toFixed(1) + 'k' }) },
		cmpTable(el) {
			const rows = U3.compare
			table(el, ['method', 'silhouette', 'Davies–B.', 'ARI risk', 'ARI true state', 'sizes'],
				rows.map(r => [r.method, r.silhouette.toFixed(3), r.davies_bouldin.toFixed(2), r.ari_at_risk.toFixed(3), r.ari_true_state.toFixed(3), r.sizes.join(' / ')]), { best: 0 })
			el.querySelector('table').style.fontSize = '16px'
		},
		dissim(el) { table(el, ['metric', 'silh.', 'ARI'], U3.dissimilarity.map(r => [r.metric, r.silhouette.toFixed(3), r.ari_at_risk.toFixed(3)])) },
		personas(el) {
			const P = U3.personas, feats = [['current_cgpa', 'CGPA'], ['attendance_percentage', 'attendance'], ['internal_assessment_average', 'internals'],
				['daily_study_hours', 'study h/day'], ['late_assignment_frequency', 'late submissions'], ['overall_academic_engagement', 'engagement'], ['active_backlogs', 'backlogs']]
			const rng = Object.fromEntries(feats.map(([f]) => { const v = P.map(p => p[f]); return [f, [Math.min(...v), Math.max(...v)]] }))
			const blurb = ['low grades, low attendance, backlogs piling up', 'decent grades, little effort: the watch list', 'high effort and high grades']
			el.innerHTML = '<div class="grid3" style="height:100%">' + P.map((p, i) => `<div class="panel pc" style="display:flex;flex-direction:column;gap:12px;border-color:${COL.s[i]}">
				<div class="row" style="align-items:baseline;justify-content:space-between"><h3 style="color:${STATE_TXT[i]};font-size:28px;white-space:nowrap">${p.name}</h3><span class="mono small" style="white-space:nowrap">${p.size} students</span></div>
				<div class="stat"><div class="v" style="font-size:72px;color:${STATE_TXT[i]}">${Math.round(100 * p.at_risk_rate)}%</div><div class="l">at risk this semester</div></div>
				<p class="small" style="margin:0;font-size:16px">${blurb[i]}</p>
				<div class="col" style="gap:9px;margin-top:auto">${feats.map(([f, lab]) => { const [a, b] = rng[f], t = b > a ? (p[f] - a) / (b - a) : 0.5
					return `<div style="display:grid;grid-template-columns:130px 1fr 54px;align-items:center;gap:10px;font-size:14px"><span class="muted">${lab}</span><span style="height:8px;border-radius:4px;background:var(--line);overflow:hidden"><i class="pb" style="display:block;height:100%;width:${8 + 92 * t}%;background:${COL.s[i]}"></i></span><span class="mono num" style="text-align:right;color:var(--ink)">${f === 'attendance_percentage' || f === 'internal_assessment_average' ? p[f].toFixed(0) : p[f].toFixed(1)}</span></div>` }).join('')}</div></div>`).join('') + '</div>'
			if (ANIM) { G.from(el.querySelectorAll('.pc'), { y: 50, opacity: 0, rotate: i => [-2, 1, 2][i], stagger: 0.18, duration: 0.8, ease: 'power3.out', delay: 0.2 }); G.from(el.querySelectorAll('.pb'), { width: 0, stagger: 0.02, duration: 0.8, delay: 0.8, ease: 'power2.out' }) }
		},
		scree(el) {
			const p = U3.pca, n = p.explained_ratio.length, [w, h] = size(el), s = svgIn(el, w, h), L = 56, B = 50, Tp = 16, R = 16
			const xs = i => L + (i + 0.5) * (w - L - R) / n, bw = (w - L - R) / n * 0.62, ys = scale(0, 1, h - B, Tp)
			axes(s, L, Tp, w - R, h - B, ys, [0, 0.25, 0.5, 0.75, 1], i => i, [], { yfmt: t => Math.round(t * 100) + '%' })
			add(s, 'line', { x1: L, x2: w - R, y1: ys(0.9), y2: ys(0.9), stroke: COL.hi, 'stroke-dasharray': '5 6' }); T(s, w - R, ys(0.9) - 12, '90 %', { anchor: 'end', fill: COL.hi, size: 13 })
			const bars = p.explained_ratio.map((v, i) => { T(s, xs(i), h - B + 20, i + 1, { anchor: 'middle', size: 12, fill: COL.muted2 }); return [add(s, 'rect', { x: xs(i) - bw / 2, y: ys(v), width: bw, height: h - B - ys(v), rx: 3, fill: i < p.k90 ? COL.d[1] : COL.line2 }), ys(v)] })
			const cum = add(s, 'path', { d: p.cumulative.map((v, i) => `${i ? 'L' : 'M'}${xs(i)},${ys(v)}`).join(' '), fill: 'none', stroke: COL.d[0], 'stroke-width': 3 })
			T(s, L + 4, h - 10, 'principal component', { size: 13, fill: COL.muted })
			if (ANIM) { bars.forEach(([b, y], i) => G.from(b, { attr: { y: h - B, height: 0 }, duration: 0.8, delay: 0.2 + i * 0.05, ease: 'power3.out' })); draw(cum, { delay: 0.9 }) }
		},
		dendro(el) {
			const dg = U3.deck.dendrogram, [w, h] = size(el), s = svgIn(el, w, h)
			const xi = dg.icoord.flat(), yi = dg.dcoord.flat(), xs = scale(Math.min(...xi), Math.max(...xi), 20, w - 20), ys = scale(0, Math.max(...yi) * 1.05, h - 24, 14)
			const order = dg.dcoord.map((d, i) => [Math.max(...d), i]).sort((a, b) => a[0] - b[0]).map(x => x[1])
			const paths = order.map(i => add(s, 'path', { d: dg.icoord[i].map((x, k) => `${k ? 'L' : 'M'}${xs(x)},${ys(dg.dcoord[i][k])}`).join(' '), fill: 'none', stroke: Math.max(...dg.dcoord[i]) > (dg.dcoord.map(d => d[1]).sort((a, b) => b - a).slice(1, 3).reduce((a, b) => a + b) / 2) ? COL.muted2 : COL.d[1], 'stroke-width': 2.2 }))
			const tops = dg.dcoord.map(d => d[1]).sort((a, b) => b - a), cutH = (tops[1] + tops[2]) / 2   // between the 2nd and 3rd highest merges
			const cut = add(s, 'line', { x1: 10, x2: w - 10, y1: ys(cutH), y2: ys(cutH), stroke: COL.hi, 'stroke-dasharray': '6 6', 'stroke-width': 2 })
			T(s, w - 12, ys(cutH) - 12, 'cut → k = 3', { anchor: 'end', fill: COL.hi, size: 14 })
			if (ANIM) { draw(paths, { d: 0.5, stagger: 0.06 }); G.from(cut, { opacity: 0, delay: 0.3 + paths.length * 0.06 }) }
		},
		factors(el) {
			const F = U3.factor_analysis.factors
			const name = items => { const t = items.map(i => i.item).join(' '); return /support|guidance/.test(t) ? 'Support' : /motivation|interest|career/.test(t) ? 'Motivation' : /satisfaction|concern|certainty/.test(t) ? 'Performance outlook' : 'Engagement' }
			el.innerHTML = `<div class="small" style="margin-bottom:14px">Cross-validated log-likelihood peaks at <b style="color:var(--ink)">${U3.factor_analysis.n_factors_best}</b> factors. The 4-factor varimax solution:</div><div class="grid4" style="height:calc(100% - 40px)">` +
				F.map(f => `<div class="panel fc"><div class="cap">Factor ${f.factor}</div><h3 style="margin-bottom:16px">${name(f.top_items)}</h3>${f.top_items.map(it => `<div style="margin:10px 0"><div class="row" style="justify-content:space-between;font-size:15px;gap:10px"><span style="color:var(--ink2)">${nice(it.item)}</span><span class="mono" style="color:${it.loading < 0 ? 'var(--coral)' : 'var(--ink)'}">${it.loading > 0 ? '+' : ''}${it.loading.toFixed(2)}</span></div><div style="height:6px;background:var(--line);border-radius:3px;margin-top:6px"><i class="fb" style="display:block;height:100%;border-radius:3px;width:${Math.abs(it.loading) * 100}%;background:${it.loading < 0 ? COL.d[2] : COL.d[0]}"></i></div></div>`).join('')}</div>`).join('') + '</div>'
			if (ANIM) { G.from(el.querySelectorAll('.fc'), { y: 40, opacity: 0, stagger: 0.14, duration: 0.7, ease: 'power3.out', delay: 0.2 }); G.from(el.querySelectorAll('.fb'), { width: 0, stagger: 0.03, duration: 0.7, delay: 0.7 }) }
		},
		trajectories(el) {
			const E = U4.examples
			el.innerHTML = `<div class="grid4" style="height:100%">${E.map((_, i) => `<div class="tj" style="height:100%;display:flex;flex-direction:column"><div class="cap" style="margin-bottom:4px">${E[i].kind}</div><div class="tc" style="flex:1;min-height:0"></div></div>`).join('')}</div>`
			el.querySelectorAll('.tc').forEach((box, i) => {
				const e = E[i]
				lineChart(box, [{ xs: e.sgpa.map((_, k) => k + 1), ys: e.sgpa, color: COL.ink2, colors: e.viterbi.map(v => COL.s[v]), r: 9 }], { ymin: 4, ymax: 10, yticks: [4, 6.5, 8, 10], left: 44, right: 14, xticks: e.sgpa.map((_, k) => k + 1) })
			})
			el.insertAdjacentHTML('beforeend', `<div class="small" style="margin-top:8px">${STATES.map((s_, i) => `<span class="dot" style="background:${COL.s[i]}"></span>${s_}&nbsp;&nbsp;&nbsp;`).join('')} · dot colour = Viterbi-decoded hidden state · gridlines at the 6.5 and 8.0 band edges</div>`)
		},
		statemachine(el) {
			const A = U4.A, pi = U4.pi, s = svgIn(el, 900, 500)
			const pos = [[160, 400], [450, 215], [740, 400]], defs = add(s, 'defs'), flows = [], nodes = []
			STATES.forEach((_, i) => { const m = add(defs, 'marker', { id: 'ar' + i, viewBox: '0 0 10 10', refX: 6, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }); add(m, 'path', { d: 'M0,0 L10,5 L0,10z', fill: COL.s[i] }) })
			for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
				const p = A[i][j]; if (p < 0.005) continue
				const wd = 1.5 + 15 * p; let d, lx, ly
				if (i === j) { const [x, y] = pos[i]; d = `M ${x - 30} ${y - 56} C ${x - 95} ${y - 200}, ${x + 95} ${y - 200}, ${x + 30} ${y - 56}`; lx = x; ly = y - 178 }
				else {
					const [x1, y1] = pos[i], [x2, y2] = pos[j], dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy), ux = dx / L, uy = dy / L, nx = -uy, ny = ux
					const cx = (x1 + x2) / 2 + 60 * nx, cy = (y1 + y2) / 2 + 60 * ny
					d = `M ${x1 + 62 * ux} ${y1 + 62 * uy} Q ${cx} ${cy} ${x2 - 66 * ux} ${y2 - 66 * uy}`; lx = 0.25 * x1 + 0.5 * cx + 0.25 * x2; ly = 0.25 * y1 + 0.5 * cy + 0.25 * y2
				}
				add(s, 'path', { d, stroke: COL.s[i], 'stroke-opacity': 0.22, 'stroke-width': wd, fill: 'none' })
				const f = add(s, 'path', { d, stroke: COL.s[i], 'stroke-width': Math.max(1.5, wd / 2.4), fill: 'none', 'stroke-dasharray': '2 14', 'marker-end': `url(#ar${i})` })
				f.dataset.speed = Math.max(0.5, 3 - 2.6 * p); flows.push(f)
				const t = T(s, lx, ly, p.toFixed(2), { anchor: 'middle', size: 16, fill: COL.ink, weight: 500 }); t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', COL.bg); t.setAttribute('stroke-width', 6)
			}
			pos.forEach(([x, y], i) => {
				const g = add(s, 'g'); nodes.push(g)
				add(g, 'circle', { cx: x, cy: y, r: 60, fill: COL.panel, stroke: COL.s[i], 'stroke-width': 3 })
				T(g, x, y - 6, STATES[i], { anchor: 'middle', size: 19, fill: COL.ink, font: 'Space Grotesk, sans-serif', weight: 700 })
				T(g, x, y + 20, `π = ${pi[i].toFixed(2)}`, { anchor: 'middle', size: 13, fill: COL.muted })
			})
			if (!PRINT && !REDUCE) flows.forEach(f => f.animate([{ strokeDashoffset: 0 }, { strokeDashoffset: -32 }], { duration: +f.dataset.speed * 1000, iterations: Infinity }))
			if (ANIM) G.from(nodes, { scale: 0, transformOrigin: 'center', transformBox: 'fill-box', stagger: 0.18, duration: 0.7, ease: 'back.out(1.8)', delay: 0.2 })
		},
		emis(el) {
			el.innerHTML = `<table class="t" style="font-size:16px"><thead><tr><th></th>${BANDS.map(b => `<th class="n">${b}</th>`).join('')}</tr></thead><tbody>${U4.B.map((r, i) => `<tr><td style="color:${STATE_TXT[i]}">${STATES[i]}</td>${r.map(v => `<td class="n" style="background:rgba(58,138,208,${(0.08 + 0.6 * v).toFixed(2)});color:var(--ink)">${v.toFixed(2)}</td>`).join('')}</tr>`).join('')}</tbody></table>`
		},
		viterbi(el) {
			const e = ex.declining || U4.examples[0]
			document.getElementById('vitTitle').textContent = `${e.kind} student · SGPA and decoded state`
			const { s, xs, ys } = lineChart(el, [{ xs: e.sgpa.map((_, i) => i + 1), ys: e.sgpa, color: COL.ink2, colors: e.viterbi.map(v => COL.s[v]), r: 13 }], { ymin: 4, ymax: 10, yticks: [4, 6.5, 8, 10], xlabel: 'semester', top: 56, right: 60 })
			e.viterbi.forEach((v, i) => T(s, xs(i + 1), ys(e.sgpa[i]) - 28, STATES[v], { anchor: 'middle', size: 13, fill: STATE_TXT[v] }))
		},
		filtered(el) {
			const e = ex.declining || U4.examples[0]
			lineChart(el, [{ xs: e.p_struggling_filtered.map((_, i) => i + 1), ys: e.p_struggling_filtered, color: COL.d[2], area: true, r: 7 }], { ymin: 0, ymax: 1, yticks: [0, 0.5, 1], xlabel: 'semester', yfmt: t => Math.round(100 * t) + '%' })
			el.insertAdjacentHTML('beforeend', `<div class="small" style="margin-top:6px">Next semester: P(Poor band) = <b style="color:var(--coral)">${(100 * e.next_band_dist[0]).toFixed(0)}%</b></div>`)
		},
		hmmbic(el) { const S = U4.selection; lineChart(el, [{ xs: S.map(r => r.K), ys: S.map(r => r.bic), color: COL.d[0], label: 'BIC' }, { xs: S.map(r => r.K), ys: S.map(r => r.aic), color: COL.d[3], label: 'AIC' }], { markX: U4.K_best_bic, markLabel: 'K = ' + U4.K_best_bic, xlabel: 'hidden states K', right: 60 }) },
		fcCat(el) { hbars(el, U4.next_semester.categorical.map(r => ({ label: r.model.replace(' (same as last)', ''), value: r.accuracy, color: r.model.startsWith('Hidden') ? COL.d[0] : COL.line2, text: (100 * r.accuracy).toFixed(1) + '%' })), { min: 0, max: 1, labelW: 200 }) },
		fcCont(el) { hbars(el, U4.next_semester.continuous.map(r => ({ label: r.model.replace(' (hmmlearn)', '').replace("Student's running mean (CGPA)", 'Running mean'), value: r.rmse, color: r.model.startsWith('Kalman') ? COL.d[0] : COL.line2, text: r.rmse.toFixed(3) })), { min: 0, labelW: 200 }) },
		early(el) { vbars(el, U4.early_warning.map((r, i) => ({ label: ['grade history\n(HMM)', 'survey\n(logistic)', 'survey\n+ HMM'][i], value: r.roc_auc, color: [COL.d[3], COL.d[0], COL.d[1]][i] })), { min: 0.5, max: 1.02 }) },
		summary(el) {
			const reg = U2.regression.table.reduce((a, b) => (b.r2 > a.r2 ? b : a)), cv = U2.classification.cv5.reduce((a, b) => (b.roc_auc_mean > a.roc_auc_mean ? b : a))
			const kal = U4.next_semester.continuous.find(r => r.model.startsWith('Kalman'))
			table(el, ['unit', 'question', 'method', 'result'], [
				['1', 'How much do two signals change the odds?', "Bayes' rule", `P(risk) ${fmt(U1.bayes.prior, 'p0')} → ${fmt(U1.bayes.chain.at(-1).p, 'p0')}`],
				['2', 'Predict end-semester SGPA', reg.model, `R² ${reg.r2.toFixed(3)}`],
				['2', 'Classify at-risk students', cv.model, `CV AUC ${cv.roc_auc_mean.toFixed(3)}`],
				['3', 'How many groups?', 'GMM + BIC', `k = ${U3.k_final}`],
				['3', 'Groups without labels', 'K-Means (scratch)', `${U3.personas.map(p => fmt(p.at_risk_rate, 'p0')).join(' · ')} at risk`],
				['3', 'How many dimensions?', 'PCA', `${U3.pca.k90} PCs → 90 %`],
				['4', 'Hidden states over semesters', 'HMM (scratch)', `K = ${U4.K_best_bic} · Viterbi ${fmt(U4.truth.viterbi_state_accuracy, 'p1')}`],
				['4', 'Next-semester SGPA', 'Kalman filter', `RMSE ${kal.rmse.toFixed(3)}`],
			], { num: [false, false, false, true] })
		},
		checks(el) {
			el.innerHTML = '<div class="col" style="gap:10px">' + Object.entries(D.meta.checks).map(([k, ok]) => `<div class="ck" style="display:flex;gap:12px;align-items:flex-start;font-size:15px"><span style="color:${ok ? 'var(--sky)' : 'var(--coral)'};font-weight:700">${ok ? '✓' : '✗'}</span><span style="color:var(--ink2)">${nice(k).replace('>=', '≥ ')}</span></div>`).join('') + '</div>'
			fade(el.querySelectorAll('.ck'), { stagger: 0.08, delay: 0.4 })
		},
	}

	/* ------------------------------------------------------------ slide-specific enters */
	function countUp(els, delay = 0.3) {
		els.forEach(el => {
			const to = parseFloat(el.dataset.to ?? el.textContent.replace(/[^0-9.\-]/g, ''))
			if (!ANIM || !isFinite(to)) return
			const txt = el.textContent, dec = (txt.split('.')[1] || '').replace(/[^0-9]/g, '').length, pct = txt.trim().endsWith('%'), o = { v: 0 }
			G.to(o, { v: to, duration: 1.4, delay, ease: 'power3.out', onUpdate: () => (el.textContent = o.v.toFixed(dec) + (pct ? '%' : '')) })
		})
	}
	const ENTER = {
		title(slide) {
			const h = slide.querySelector('#heroTitle')
			if (!h.dataset.split) { h.innerHTML = [...h.textContent].map(c => `<span class="ch">${c}</span>`).join(''); h.dataset.split = 1 }
			if (ANIM) G.from(h.querySelectorAll('.ch'), { yPercent: 110, rotate: 8, opacity: 0, stagger: 0.05, duration: 0.9, ease: 'power4.out', delay: 0.15 })
		},
		kmeans: () => kmeans.play(),
		findings() {
			const ul = document.getElementById('findList'); if (ul.dataset.done) return; ul.dataset.done = 1
			const P = U3.personas, ew = U4.early_warning
			const items = [
				`Internal marks (r = ${U1.corr_with_target.internal_assessment_average.toFixed(2)}) and attendance (r = ${U1.corr_with_target.attendance_percentage.toFixed(2)}) are the strongest protective signals.`,
				`Linear models already reach CV AUC ≈ ${U2.classification.cv5[0].roc_auc_mean.toFixed(2)}; the Bayesian model adds a credible band to every prediction.`,
				`Clustering finds a ${P[1].name.toLowerCase()} group: decent grades, low effort, only ${fmt(P[1].at_risk_rate, 'p0')} at risk today, but a natural watch list.`,
				`Struggling is sticky: P(stay) = ${U4.A[0][0].toFixed(2)}, the case for intervening early.`,
				`Grade history alone predicts this semester's risk with AUC ${ew[0].roc_auc.toFixed(2)}; the survey lifts it to ${ew[1].roc_auc.toFixed(2)}.`,
			]
			ul.innerHTML = items.map(t => `<li>${t}</li>`).join('')
			fade(ul.querySelectorAll('li'), { stagger: 0.12, delay: 0.4 })
		},
	}

	/* ------------------------------------------------------------ K-Means replay */
	const kmeans = (() => {
		const cv = document.getElementById('kmCanvas'), ctx = cv.getContext('2d'), P = U3.deck.points, TR = U3.deck.kmeans_trace
		const W = cv.width, H = cv.height, pad = 60
		const xs = scale(Math.min(...P.map(p => p[0])) - 0.3, Math.max(...P.map(p => p[0])) + 0.3, pad, W - pad)
		const ys = scale(Math.min(...P.map(p => p[1])) - 0.3, Math.max(...P.map(p => p[1])) + 0.3, H - pad, pad)
		const st = { it: 0, t: 1, cent: TR[0].centroids.map(c => [...c]), trail: [[], [], []] }
		let raf = 0, timer = 0
		const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16))
		const S = COL.s.map(rgb)
		function frame() {
			ctx.clearRect(0, 0, W, H)
			ctx.strokeStyle = COL.line; ctx.lineWidth = 1
			for (let gx = 0; gx <= 10; gx++) { const x = pad + gx * (W - 2 * pad) / 10; ctx.beginPath(); ctx.moveTo(x, pad); ctx.lineTo(x, H - pad); ctx.stroke() }
			for (let gy = 0; gy <= 6; gy++) { const y = pad + gy * (H - 2 * pad) / 6; ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(W - pad, y); ctx.stroke() }
			const labs = TR[st.it].labels
			P.forEach((p, i) => {
				const c = S[labs[i]]
				ctx.beginPath(); ctx.arc(xs(p[0]), ys(p[1]), 7, 0, 7)
				ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.35 + 0.55 * st.t})`; ctx.fill()
				if (p[3]) { ctx.strokeStyle = 'rgba(236,235,228,.55)'; ctx.lineWidth = 1.5; ctx.stroke() }
			})
			st.cent.forEach((c, k) => {
				const tr = st.trail[k]; ctx.strokeStyle = COL.ink; ctx.lineWidth = 2; ctx.setLineDash([6, 6]); ctx.beginPath()
				tr.forEach((q, j) => (j ? ctx.lineTo(xs(q[0]), ys(q[1])) : ctx.moveTo(xs(q[0]), ys(q[1])))); ctx.lineTo(xs(c[0]), ys(c[1])); ctx.stroke(); ctx.setLineDash([])
				ctx.beginPath(); ctx.arc(xs(c[0]), ys(c[1]), 20, 0, 7); ctx.fillStyle = COL.bg; ctx.fill(); ctx.lineWidth = 5; ctx.strokeStyle = COL.s[k]; ctx.stroke()
				ctx.beginPath(); ctx.moveTo(xs(c[0]) - 9, ys(c[1])); ctx.lineTo(xs(c[0]) + 9, ys(c[1])); ctx.moveTo(xs(c[0]), ys(c[1]) - 9); ctx.lineTo(xs(c[0]), ys(c[1]) + 9); ctx.strokeStyle = COL.ink; ctx.lineWidth = 3; ctx.stroke()
			})
			ctx.font = '500 26px IBM Plex Mono, monospace'; ctx.fillStyle = COL.muted
			ctx.fillText('PC1 →', W - pad - 90, H - 18); ctx.save(); ctx.translate(26, pad + 70); ctx.rotate(-Math.PI / 2); ctx.fillText('PC2 →', 0, 0); ctx.restore()
			ctx.fillStyle = COL.ink2; ctx.font = '500 24px IBM Plex Sans, sans-serif'
			STATES.forEach((_, k) => { ctx.fillStyle = COL.s[k]; ctx.fillRect(pad + k * 290, 20, 18, 18); ctx.fillStyle = COL.ink2; ctx.fillText(U3.personas[k].name, pad + 28 + k * 290, 37) })
			document.getElementById('kmIter').textContent = st.it + 1
			document.getElementById('kmJ').textContent = TR[st.it].J.toFixed(0)
		}
		function step() {
			if (st.it >= TR.length - 1) return
			const from = st.cent.map(c => [...c]); st.trail.forEach((t, k) => t.push(from[k]))
			st.it++; const to = TR[st.it].centroids, t0 = performance.now(), dur = st.it < 6 ? 700 : 260
			cancelAnimationFrame(raf)
			const tick = now => {
				const u = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - u, 3)
				st.cent = from.map((c, k) => [c[0] + (to[k][0] - c[0]) * e, c[1] + (to[k][1] - c[1]) * e]); st.t = Math.min(1, 0.4 + u)
				frame(); if (u < 1) raf = requestAnimationFrame(tick); else timer = setTimeout(step, st.it < 6 ? 350 : 80)
			}
			raf = requestAnimationFrame(tick)
		}
		function reset() { clearTimeout(timer); cancelAnimationFrame(raf); st.it = 0; st.t = 0.4; st.cent = TR[0].centroids.map(c => [...c]); st.trail = [[], [], []] }
		document.getElementById('kmReplay').addEventListener('click', e => { e.stopPropagation(); reset(); frame(); timer = setTimeout(step, 400) })
		return {
			play() { reset(); frame(); if (PRINT || REDUCE) { st.it = TR.length - 1; st.cent = TR[st.it].centroids; st.t = 1; frame() } else timer = setTimeout(step, 700) },
			final() { reset(); st.it = TR.length - 1; st.cent = TR[st.it].centroids; st.t = 1; frame() },
		}
	})()

	/* ------------------------------------------------------------ background: chalk dust grid */
	;(function bg() {
		const cv = document.getElementById('bg'); if (PRINT) return
		const ctx = cv.getContext('2d'), dots = [], m = { x: -999, y: -999 }; let w, h
		const build = () => {
			const dpr = Math.min(2, devicePixelRatio || 1); w = innerWidth; h = innerHeight; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
			dots.length = 0; for (let y = 12; y < h; y += 26) for (let x = 12; x < w; x += 26) dots.push([x, y, 0, 0])
		}
		addEventListener('resize', build); addEventListener('mousemove', e => { m.x = e.clientX; m.y = e.clientY }, { passive: true })
		build()
		;(function loop() {
			ctx.clearRect(0, 0, w, h)
			for (const d of dots) {
				const dx = d[0] - m.x, dy = d[1] - m.y, dist = Math.hypot(dx, dy), f = dist < 160 ? 1 - dist / 160 : 0
				d[2] += ((f ? dx / dist * f * 10 : 0) - d[2]) * 0.12; d[3] += ((f ? dy / dist * f * 10 : 0) - d[3]) * 0.12
				ctx.fillStyle = f ? `rgba(227,195,90,${0.25 + 0.6 * f})` : 'rgba(236,235,228,.07)'
				ctx.beginPath(); ctx.arc(d[0] + d[2], d[1] + d[3], f ? 1.4 + 1.6 * f : 1.2, 0, 7); ctx.fill()
			}
			if (!REDUCE) requestAnimationFrame(loop)
		})()
	})()

	/* ------------------------------------------------------------ stage, navigation, transitions */
	const stage = document.getElementById('stage'), slides = [...document.querySelectorAll('.slide')]
	const built = new Set()
	function fit() { if (PRINT) return; const s = Math.min(innerWidth / 1600, innerHeight / 900); stage.style.transform = `scale(${s})` }
	addEventListener('resize', fit); fit()

	function build(slide) {
		if (built.has(slide)) return
		slide.querySelectorAll('[data-chart]').forEach(el => { try { CHARTS[el.dataset.chart]?.(el) } catch (err) { console.error(el.dataset.chart, err) } })
		built.add(slide)
	}
	function rebuild(slide) { built.delete(slide); build(slide) }

	const wipe = document.createElement('div')
	wipe.style.cssText = 'position:absolute;inset:0;z-index:15;pointer-events:none;background:var(--panel2);clip-path:inset(0 100% 0 0);display:grid;place-items:center'
	wipe.innerHTML = '<div style="font:700 120px var(--display);letter-spacing:-.05em;color:transparent;-webkit-text-stroke:1.5px var(--hi)"></div>'
	stage.appendChild(wipe)

	let cur = -1, busy = false
	function show(i, dir = 1) {
		i = Math.max(0, Math.min(slides.length - 1, i))
		if (i === cur || busy) return
		const prev = slides[cur], next = slides[i], sectionChange = prev && prev.dataset.section !== next.dataset.section
		const reveal = () => {
			if (prev) prev.classList.remove('active')
			next.classList.add('active')
			rebuild(next)
			if (ANIM) G.fromTo(next.querySelectorAll('.rv'), { y: 34 * dir, opacity: 0, filter: 'blur(8px)' }, { y: 0, opacity: 1, filter: 'blur(0px)', duration: 0.8, stagger: 0.07, ease: 'power3.out', clearProps: 'filter,transform' })
			next.querySelectorAll('.cu').forEach(el => countUp([el], 0.5))
			ENTER[next.dataset.enter]?.(next)
			cur = i
			chrome()
			try { history.replaceState(null, '', '#s' + (i + 1)) } catch (e) {}
		}
		if (!ANIM || !prev) { reveal(); return }
		busy = true
		if (sectionChange) {
			wipe.firstChild.textContent = next.dataset.section
			G.timeline({ onComplete: () => (busy = false) })
				.fromTo(wipe, { clipPath: dir > 0 ? 'inset(0 100% 0 0)' : 'inset(0 0 0 100%)' }, { clipPath: 'inset(0 0% 0 0%)', duration: 0.5, ease: 'power3.inOut' })
				.add(reveal)
				.fromTo(wipe.firstChild, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.3 }, '<-0.1')
				.to(wipe, { clipPath: dir > 0 ? 'inset(0 0 0 100%)' : 'inset(0 100% 0 0)', duration: 0.55, ease: 'power3.inOut', delay: 0.25 })
		} else {
			G.to(prev.querySelectorAll('.rv, .body, .chart'), { opacity: 0, y: -24 * dir, duration: 0.3, ease: 'power2.in', onComplete: () => { G.set(prev.querySelectorAll('.rv, .body, .chart'), { clearProps: 'all' }); reveal(); busy = false } })
		}
	}
	const next = () => show(cur + 1, 1), prev = () => show(cur - 1, -1)

	function chrome() {
		const s = slides[cur]
		document.getElementById('secLabel').textContent = s.dataset.section
		document.getElementById('counter').textContent = `${String(cur + 1).padStart(2, '0')} / ${slides.length}`
		document.getElementById('progress').style.width = `${(100 * (cur + 1)) / slides.length}%`
		const n = document.getElementById('notes')
		n.innerHTML = `<div class="who">Speaker · ${s.dataset.speaker || ''} <span style="color:var(--muted2)">(suggested)</span></div>${s.dataset.notes || ''}`
	}

	/* ------------------------------------------------------------ overview + keys */
	const ov = document.getElementById('ov')
	function toggleOverview(force) {
		const on = force ?? ov.hidden
		if (on) {
			ov.innerHTML = slides.map((s, i) => `<button type="button" data-i="${i}" class="${i === cur ? 'cur' : ''}"><small>${String(i + 1).padStart(2, '0')} · ${s.dataset.section}</small>${s.dataset.title}</button>`).join('')
			ov.hidden = false; ov.querySelector('.cur')?.focus()
		} else ov.hidden = true
	}
	ov.addEventListener('click', e => { const b = e.target.closest('button'); if (b) { toggleOverview(false); show(+b.dataset.i, +b.dataset.i > cur ? 1 : -1) } })
	addEventListener('keydown', e => {
		if (e.target.closest('input,textarea')) return
		const k = e.key
		if (['ArrowRight', 'PageDown', ' ', 'Enter'].includes(k) && ov.hidden) { e.preventDefault(); next() }
		else if (['ArrowLeft', 'PageUp', 'Backspace'].includes(k) && ov.hidden) { e.preventDefault(); prev() }
		else if (k === 'Home') show(0, -1)
		else if (k === 'End') show(slides.length - 1, 1)
		else if (k === 'n' || k === 'N' || k === 's' || k === 'S') { const n = document.getElementById('notes'); n.hidden = !n.hidden }
		else if (k === 'o' || k === 'O' || (k === 'Escape' && !ov.hidden)) toggleOverview()
		else if (k === 'f' || k === 'F') { (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.())?.catch?.(() => {}) }
	})
	document.getElementById('viewport').addEventListener('click', e => { if (e.target.closest('button,a')) return; (e.clientX > innerWidth * 0.35 ? next : prev)() })
	let tx = 0
	addEventListener('touchstart', e => (tx = e.touches[0].clientX), { passive: true })
	addEventListener('touchend', e => { const dx = e.changedTouches[0].clientX - tx; if (Math.abs(dx) > 50) (dx < 0 ? next : prev)() }, { passive: true })

	/* ------------------------------------------------------------ boot */
	if (PRINT) {
		document.body.classList.add('print')
		const only = +(location.hash.match(/^#s(\d+)$/)?.[1] || 0)   // ?print#s7 → just that slide, static
		slides.forEach((s, i) => { if (only && i !== only - 1) { s.remove(); return } s.classList.add('active'); build(s); ENTER[s.dataset.enter]?.(s) })
		if (document.getElementById('kmCanvas')) kmeans.final()
		return
	}
	const start = +(location.hash.match(/^#s(\d+)$/)?.[1] || 1) - 1
	const go = () => show(start, 1)
	document.fonts?.ready ? document.fonts.ready.then(go) : go()
})()
