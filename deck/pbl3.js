/* EarlyDrop PBL-III deck (Unit 5 · combining models). Same engine and look as deck.js; every number comes from
 * window.EARLYDROP.unit5 (deck-data.js, written by run_all.py). Charts are hand-built SVG. */
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

	/* ------------------------------------------------------------ chart registry (PBL-III · Unit 5) */
	const U2 = D.unit2, U5 = D.unit5
	const FAMC = { single: COL.line2, bagging: COL.d[1], boosting: COL.d[0], averaging: COL.d[3], combining: COL.d[4] }
	const SHORT = { 'internal assessment average': 'internal marks', 'attendance percentage': 'attendance %', 'previous sem sgpa': 'prev. SGPA',
		'current cgpa': 'CGPA', 'programme completion confidence': 'confidence', 'academic performance concern': 'concern', 'active backlogs': 'backlogs',
		'daily study hours': 'study hours', 'academic motivation': 'motivation', 'overall academic engagement': 'engagement',
		'late assignment frequency': 'late submissions', 'academic workload difficulty': 'workload', 'assignment completion consistency': 'assignment consistency' }
	const short = f => SHORT[nice(f)] || nice(f)
	const cvRow = prefix => U5.compare_cv.find(r => r.model.startsWith(prefix))
	const lg = v => Math.log10(v)
	const hex = c => { c = c.replace('#', ''); return [0, 2, 4].map(i => parseInt(c.slice(i, i + 2), 16)) }
	const mix = (a, b, t) => { const A = hex(a), B = hex(b); return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})` }
	const legend = items => `<div class="small" style="display:flex;gap:18px;flex-wrap:wrap;margin-top:8px">${items.map(([c, t, dash]) => `<span><span class="dot" style="${dash ? `border:2px dashed ${c};background:none` : `background:${c}`}"></span>${t}</span>`).join('')}</div>`

	const sub = (el, room = 34) => { el.innerHTML = `<div style="height:calc(100% - ${room}px)"></div>`; return el.firstChild }  // chart + legend row

	const CHARTS = {
		tree(el) {
			const R = JSON.parse(JSON.stringify(U5.cart.tree_depth3)), [w, h] = size(el), s = svgIn(el, w, h)
			let leaf = 0; const nodes = [], edges = [], boxes = []
			const walk = (n, d) => { n.d = d; if (n.left) { walk(n.left, d + 1); walk(n.right, d + 1); n.x = (n.left.x + n.right.x) / 2 } else n.x = leaf++; nodes.push(n) }
			walk(R, 0)
			const maxD = Math.max(...nodes.map(n => n.d)), xs = scale(-0.5, leaf - 0.5, 0, w), ys = d => 34 + d * (h - 76) / maxD
			nodes.forEach(n => n.left && [n.left, n.right].forEach((c, k) => {
				edges.push(add(s, 'path', { d: `M${xs(n.x)},${ys(n.d) + 26} C${xs(n.x)},${ys(n.d) + 70} ${xs(c.x)},${ys(c.d) - 70} ${xs(c.x)},${ys(c.d) - 26}`, fill: 'none', stroke: COL.line2, 'stroke-width': 2 }))
				T(s, (xs(n.x) + xs(c.x)) / 2 + (k ? 10 : -10), (ys(n.d) + ys(c.d)) / 2, k ? 'no' : 'yes', { anchor: k ? 'start' : 'end', size: 12, fill: COL.muted2 })
			}))
			nodes.forEach(n => {
				const g = add(s, 'g'), inner = !!n.left, bw = inner ? 232 : 150, bh = 54, x = xs(n.x) - bw / 2, y = ys(n.d) - 27
				add(g, 'rect', { x, y, width: bw, height: bh, rx: 8, fill: COL.panel, stroke: inner ? COL.line2 : (n.p > 0.5 ? COL.coral : COL.sky), 'stroke-width': inner ? 1.5 : 2.5 })
				add(g, 'rect', { x: x + 10, y: y + bh - 10, width: bw - 20, height: 4, rx: 2, fill: COL.line })
				add(g, 'rect', { x: x + 10, y: y + bh - 10, width: Math.max(1, (bw - 20) * n.p), height: 4, rx: 2, fill: COL.coral })
				T(g, xs(n.x), y + 16, inner ? `${short(n.feature)} ≤ ${n.threshold}` : `${(100 * n.p).toFixed(0)}% at risk`, { anchor: 'middle', size: inner ? 15 : 15, fill: COL.ink, weight: 600, font: 'IBM Plex Sans, sans-serif' })
				T(g, xs(n.x), y + 33, inner ? `n = ${n.n} · gini ${n.gini.toFixed(2)}` : `n = ${n.n}`, { anchor: 'middle', size: 12, fill: COL.muted })
				boxes.push(g)
			})
			draw(edges, { d: 0.7, stagger: 0.04 }); pop(boxes, { stagger: 0.05 })
		},
		depth(el) {
			const d = U5.cart.depth_curve, x = d.map(r => r.depth)
			lineChart(sub(el, 34), [{ xs: x, ys: d.map(r => r.train_acc), color: COL.d[0] }, { xs: x, ys: d.map(r => r.cv_acc), color: COL.d[1] },
				{ xs: x, ys: d.map(r => r.test_acc), color: COL.d[2], dash: '6 6' }], { xticks: [1, 2, 4, 6, 8, 10, 12], xlabel: 'max depth', right: 24, yfmt: t => (100 * t).toFixed(0) + '%' })
			el.insertAdjacentHTML('beforeend', legend([[COL.d[0], 'train'], [COL.d[1], '5-fold CV'], [COL.d[2], 'held-out test', 1]]))
		},
		oob(el) {
			const b = U5.forest.oob_curve_bagging, r = U5.forest.oob_curve_rf
			lineChart(sub(el, 34), [{ xs: b.map(p => lg(p.n_trees)), ys: b.map(p => p.oob_error), color: COL.d[0] }, { xs: r.map(p => lg(p.n_trees)), ys: r.map(p => p.oob_error), color: COL.d[1] }],
				{ xticks: [0, 1, 2, lg(200)], xfmt: t => Math.round(10 ** t), xlabel: 'number of trees (log scale)', right: 24, yfmt: t => (100 * t).toFixed(0) + '%' })
			el.insertAdjacentHTML('beforeend', legend([[COL.d[0], 'bagging (all 15 features per split)'], [COL.d[1], 'random forest (3 random features per split)']]))
		},
		bv(el) {
			const rows = U5.bias_variance.rows, [w, h] = size(el), s = svgIn(el, w, h), LW = 230, max = Math.max(...rows.map(r => r.brier)) * 1.12
			const xs = scale(0, max, LW, w - 70), rh = (h - 30) / rows.length, bars = []
			rows.forEach((r, i) => {
				const cy = 8 + i * rh + rh / 2, hl = r.model.startsWith('Random') || r.model.startsWith('CART (full')
				T(s, LW - 12, cy, r.model, { anchor: 'end', size: 15, fill: hl ? COL.ink : COL.ink2, font: 'IBM Plex Sans, sans-serif', weight: hl ? 600 : 400 })
				bars.push([add(s, 'rect', { x: LW, y: cy - 12, width: xs(r.bias2) - LW, height: 24, rx: 3, fill: COL.d[0] }), LW])
				bars.push([add(s, 'rect', { x: xs(r.bias2), y: cy - 12, width: Math.max(1, xs(r.brier) - xs(r.bias2)), height: 24, rx: 3, fill: COL.d[2] }), xs(r.bias2)])
				T(s, xs(r.brier) + 10, cy, r.brier.toFixed(3), { size: 14, fill: COL.ink, weight: 500 })
			})
			if (ANIM) bars.forEach(([b], i) => G.from(b, { attr: { width: 0 }, duration: 0.9, ease: 'power3.out', delay: 0.2 + i * 0.04 }))
			el.insertAdjacentHTML('beforeend', legend([[COL.d[0], 'bias²'], [COL.d[2], 'variance']]))
		},
		surfaces(el) {
			const S = U5.surfaces, names = Object.keys(S.models)
			el.innerHTML = `<div class="grid4" style="height:100%;gap:22px">${names.map((n, i) => `<div class="col" style="gap:6px;min-height:0"><div class="cap" style="margin:0">${n} · train ${fmt(S.models[n].train_acc, 'p0')}</div><div class="sf" data-i="${i}" style="flex:1;min-height:0"></div></div>`).join('')}</div>`
			el.querySelectorAll('.sf').forEach(box => {
				const m = S.models[names[+box.dataset.i]], [w, h] = size(box), s = svgIn(box, w, h), L = 34, B = 34
				const nx = S.gx.length, ny = S.gy.length, dx = S.gx[1] - S.gx[0], dy = S.gy[1] - S.gy[0]
				const xs = scale(S.gx[0] - dx / 2, S.gx[nx - 1] + dx / 2, L, w), ys = scale(S.gy[0] - dy / 2, S.gy[ny - 1] + dy / 2, h - B, 0)
				const cells = add(s, 'g'), cw = xs(S.gx[1]) - xs(S.gx[0]) + 0.6, ch = ys(S.gy[0]) - ys(S.gy[1]) + 0.6
				m.p.forEach((p, k) => { const i = k % nx, j = Math.floor(k / nx); add(cells, 'rect', { x: xs(S.gx[i]) - cw / 2, y: ys(S.gy[j]) - ch / 2, width: cw, height: ch, fill: mix('#1b2d26', '#cf4b44', Math.min(1, p * 1.1)) }) })
				const pts = add(s, 'g')
				S.points.forEach(([a, b, c]) => { if (a < S.gx[0] || a > S.gx[nx - 1] || b < S.gy[0] || b > S.gy[ny - 1]) return
					add(pts, 'circle', { cx: xs(a), cy: ys(b), r: c ? 3.2 : 2.2, fill: c ? '#f3b2a8' : COL.ink, opacity: c ? 0.95 : 0.35, stroke: c ? COL.bg : 'none', 'stroke-width': 0.8 }) })
				T(s, (L + w) / 2, h - 10, 'attendance %', { anchor: 'middle', size: 12, fill: COL.muted })
				T(s, L - 6, h - B - 4, Math.round(S.gy[0]), { anchor: 'end', size: 11, fill: COL.muted2 }); T(s, L - 6, 8, Math.round(S.gy[ny - 1]), { anchor: 'end', size: 11, fill: COL.muted2 })
				T(s, L, h - B + 12, Math.round(S.gx[0]), { size: 11, fill: COL.muted2 }); T(s, w, h - B + 12, Math.round(S.gx[nx - 1]), { anchor: 'end', size: 11, fill: COL.muted2 })
				const yl = T(s, 12, (h - B) / 2, 'internal marks %', { anchor: 'middle', size: 12, fill: COL.muted }); yl.setAttribute('transform', `rotate(-90 12 ${(h - B) / 2})`)
				fade([cells], { delay: 0.3 }); fade([pts], { delay: 0.9 })
			})
		},
		ada(el) {
			const c = U5.adaboost.curve, x = c.map(r => lg(r.m))
			lineChart(sub(el, 34), [{ xs: x, ys: c.map(r => r.exp_loss), color: COL.muted, dash: '6 6', markers: false },
				{ xs: x, ys: c.map(r => r.train_error), color: COL.d[0], markers: false },
				{ xs: x, ys: c.map(r => r.test_error), color: COL.d[1], markers: false }],
				{ ymin: 0, ymax: 0.6, yticks: [0, 0.1, 0.2, 0.3, 0.4, 0.5], xticks: [0, 1, 2, lg(200)], xfmt: t => Math.round(10 ** t), xlabel: 'boosting round m (log scale)', right: 24 })
			el.insertAdjacentHTML('beforeend', legend([[COL.d[0], 'train error'], [COL.d[1], 'test error'], [COL.muted, 'exponential loss (upper bound on train error)', 1]]))
		},
		shrink(el) {
			const S = U5.gboost.shrinkage
			lineChart(sub(el, 34), S.map((g, i) => ({ xs: g.curve.map(r => r.m), ys: g.curve.map(r => Math.min(0.75, r.test)), color: [COL.d[2], COL.d[0], COL.d[1], COL.d[4]][i], markers: false })),
				{ ymin: 0.2, ymax: 0.75, yticks: [0.25, 0.4, 0.55, 0.7], xticks: [1, 50, 100, 150, 200, 250, 300], xlabel: 'iterations', right: 24 })
			el.insertAdjacentHTML('beforeend', legend(S.map((g, i) => [[COL.d[2], COL.d[0], COL.d[1], COL.d[4]][i], 'ν = ' + g.learning_rate])))
		},
		bmaInc(el) {
			const I = U5.bma.inclusion
			hbars(el, Object.entries(I).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: k, value: v, color: v > 0.5 ? COL.d[3] : COL.line2, text: v.toFixed(2) })), { min: 0, max: 1, labelW: 150, f: 'f2' })
		},
		bmaTop(el) {
			table(el, ['feature blocks in the model', 'p(M | D)'], U5.bma.top_models.slice(0, 5).map(m => [m.blocks.join(' + ') || 'intercept only', fmt(m.posterior, 'p1')]), { best: 0 })
		},
		bmaConc(el) {
			const c = U5.bma.concentration
			lineChart(el, [{ xs: c.map(r => lg(r.n)), ys: c.map(r => r.top_posterior), color: COL.d[3], label: 'top model' }],
				{ ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: c.map(r => lg(r.n)), xfmt: t => Math.round(10 ** t), xlabel: 'training students', right: 100, yfmt: t => (100 * t).toFixed(0) + '%' })
		},
		gam(el) {
			const S = U5.gam.shapes.slice(0, 6)
			el.innerHTML = `<div class="grid3" style="height:100%;gap:20px">${S.map((s, i) => `<div class="panel" style="padding:14px 16px;display:flex;flex-direction:column"><div class="cap" style="margin:0 0 4px">${short(s.feature)}</div><div class="g" data-i="${i}" style="flex:1;min-height:0"></div></div>`).join('')}</div>`
			el.querySelectorAll('.g').forEach(box => {
				const s = S[+box.dataset.i], lo = Math.min(...s.x), hi = Math.max(...s.x)
				lineChart(box, [{ xs: s.x, ys: s.f, color: COL.d[[2, 1, 0, 3, 4, 2][+box.dataset.i]], markers: false, width: 3.5 }],
					{ xticks: [lo, hi], xfmt: t => +t.toFixed(1), left: 46, right: 14, yfmt: t => (t > 0 ? '+' : '') + t })
			})
		},
		stackW(el) {
			const Wt = U5.stacking.meta_weights, A = U5.stacking.base_auc_oof, L = { lr: 'logistic', svm: 'SVM (RBF)', nb: 'naive Bayes', rf: 'random forest', gb: 'gradient boosting' }
			hbars(el, Object.entries(Wt).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: `${L[k]} · AUC ${A[k].toFixed(3)}`, value: v, color: v > 1 ? COL.d[4] : COL.line2, text: v.toFixed(2) })), { labelW: 270, f: 'f2' })
		},
		corr(el) {
			const S = U5.stacking, L = { lr: 'LR', svm: 'SVM', nb: 'NB', rf: 'RF', gb: 'GB' }
			el.innerHTML = `<table class="t" style="font-size:16px"><thead><tr><th></th>${S.bases.map(b => `<th class="n">${L[b]}</th>`).join('')}</tr></thead><tbody>${S.base_corr.map((r, i) => `<tr><td>${L[S.bases[i]]}</td>${r.map(v => `<td class="n" style="background:rgba(47,154,116,${(Math.max(0, v - 0.6) * 1.6).toFixed(2)});color:var(--ink)">${v.toFixed(2)}</td>`).join('')}</tr>`).join('')}</tbody></table>`
			fade(el.querySelectorAll('tbody tr'), { delay: 0.35, stagger: 0.06 })
		},
		verify(el) {
			const c4 = U5.cart.scratch_vs_sklearn_depth4, cf = U5.cart.scratch_vs_sklearn_full, F = U5.forest, A = U5.adaboost.scratch_vs_sklearn, B = U5.gboost.scratch_vs_sklearn
			table(el, ['scratch model', 'checked against', 'evidence', 'result'], [
				['CART, depth 4', 'DecisionTreeClassifier', `same root split: ${short(c4.root_feature)} ≤ ${c4.root_threshold}`, `${fmt(c4.test_agreement, 'p0')} same test predictions`],
				['CART, full depth', 'DecisionTreeClassifier', `${cf.leaves_scratch} vs ${cf.leaves_sklearn} leaves · depth ${cf.depth_scratch} vs ${cf.depth_sklearn}`, `${fmt(cf.test_agreement, 'p0')} agreement (ties break differently)`],
				['Random forest, 200 trees', 'RandomForestClassifier', `OOB accuracy ${fmt(F.oob_acc_rf, 'p1')} vs ${fmt(F.oob_acc_sklearn_rf, 'p1')}`, `test AUC ${F.test_auc_scratch_rf.toFixed(3)} vs ${F.test_auc_sklearn_rf.toFixed(3)}`],
				['AdaBoost, 200 stumps', 'AdaBoostClassifier (SAMME)', `decision-function correlation ${A.decision_corr.toFixed(3)}`, `${fmt(A.test_agreement, 'p0')} same predictions`],
				['Gradient boosting, 150 trees', 'GradientBoostingClassifier', `max |Δp| = ${B.max_abs_proba_diff.toFixed(4)}`, `test AUC ${B.test_auc_scratch.toFixed(3)} = ${B.test_auc_sklearn.toFixed(3)}`],
				['Bayesian model averaging', 'Laplace evidence (Bishop 4.137)', `${U5.bma.n_models} Bayesian logistic models`, `test AUC ${U5.bma.test.BMA.roc_auc.toFixed(3)}`],
			], { num: [false, false, false, false] })
		},
		cmp(el) {
			const R = U5.compare_cv
			hbars(el, R.map(r => ({ label: r.model, value: r.auc_fold_mean, err: r.auc_fold_std, color: FAMC[r.family], text: r.auc_fold_mean.toFixed(3), labelColor: r.family === 'single' ? COL.muted : COL.ink2 })),
				{ min: 0.75, max: 1.0, labelW: 360, rowH: 32, barH: 18, labelSize: 14 })
			el.insertAdjacentHTML('beforeend', legend([[COL.line2, 'single model'], [COL.d[1], 'bagging'], [COL.d[0], 'boosting'], [COL.d[3], 'model averaging'], [COL.d[4], 'voting / stacking']]))
		},
		sig(el) {
			table(el, ['vs logistic regression', 'AUC (15 folds)', 'Δ', 'p'], U5.significance.filter(r => !r.model.startsWith('Logistic')).map(r =>
				[r.model.replace(' (sklearn, 300)', '').replace(' (sklearn)', '').replace(' (sklearn, 200 stumps)', '').replace(' (LR meta-learner)', '').replace(' (LR + SVM + RF + GB)', ''), `${r.auc_mean.toFixed(3)} ± ${r.auc_std.toFixed(3)}`, fmt(r.diff_vs_lr, 's3'), r.p_value.toFixed(2)]))
		},
		roc(el) {
			const R = U5.roc, names = Object.keys(R), cols = [COL.muted, COL.d[2], COL.d[1], COL.d[0], COL.d[4], COL.d[3]]
			lineChart(sub(el, 58), names.map((n, i) => ({ xs: R[n].fpr, ys: R[n].tpr, color: cols[i % cols.length], markers: false, width: i === 0 ? 3.5 : 2.5, dash: i === 0 ? '7 6' : '' })),
				{ ymin: 0, ymax: 1, yticks: [0, 0.25, 0.5, 0.75, 1], xticks: [0, 0.25, 0.5, 0.75, 1], xlabel: 'false positive rate', left: 50 })
			el.insertAdjacentHTML('beforeend', legend(names.map((n, i) => [cols[i % cols.length], `${n.replace(' (Unit 2)', '').replace(' (scratch, full depth)', '').replace(' (sklearn, 300)', '').replace(' (sklearn)', '').replace(' (LR meta-learner)', '').replace(' (scratch)', '')} ${R[n].auc.toFixed(3)}`])))
		},
		imp(el) {
			const R = U5.importance.slice(0, 7), mx = k => Math.max(...U5.importance.map(r => r[k]))
			const bar = (v, m, c) => `<span style="display:inline-block;vertical-align:middle;height:10px;width:${Math.max(2, 90 * v / m).toFixed(0)}px;background:${c};border-radius:2px;margin-right:8px"></span>${v.toFixed(3)}`
			table(el, ['feature', 'GB permutation (ΔAUC)', 'RF impurity', '|logistic w|'], R.map(r => [short(r.feature), bar(r.gb_permutation, mx('gb_permutation'), COL.d[0]), bar(r.rf_impurity, mx('rf_impurity'), COL.d[1]), bar(r.lr_abs_coef, mx('lr_abs_coef'), COL.d[3])]), { num: [false, false, false, false] })
		},
		thr(el) {
			table(el, ['threshold', 'recall', 'precision', 'F1'], U5.thresholds_best_ensemble.map(r => [r.threshold.toFixed(1), fmt(r.recall, 'p0'), fmt(r.precision, 'p0'), r.f1.toFixed(2)]), { best: 2 })
		},
		checks(el) {
			el.innerHTML = '<div class="col" style="gap:9px">' + Object.entries(D.meta.checks).map(([k, ok]) => `<div class="ck" style="display:flex;gap:12px;align-items:flex-start;font-size:15px"><span style="color:${ok ? 'var(--sky)' : 'var(--coral)'};font-weight:700">${ok ? '✓' : '✗'}</span><span style="color:var(--ink2)">${nice(k).replace('>=', '≥ ')}</span></div>`).join('') + '</div>'
			fade(el.querySelectorAll('.ck'), { stagger: 0.06, delay: 0.4 })
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
			if (!h.dataset.split) { h.innerHTML = [...h.textContent].map(c => `<span class="ch">${c === ' ' ? '&nbsp;' : c}</span>`).join(''); h.dataset.split = 1 }
			if (ANIM) G.from(h.querySelectorAll('.ch'), { yPercent: 110, rotate: 8, opacity: 0, stagger: 0.05, duration: 0.9, ease: 'power4.out', delay: 0.15 })
		},
		findings() {
			const ul = document.getElementById('findList'); if (ul.dataset.done) return; ul.dataset.done = 1
			const tree = cvRow('CART (scratch'), rf = cvRow('Random Forest (scratch'), lr = cvRow('Logistic'), best = U5.compare_cv[0]
			const bv = Object.fromEntries(U5.bias_variance.rows.map(r => [r.model, r]))
			const sig = U5.significance.filter(r => !r.model.startsWith('Logistic')), minp = Math.min(...sig.map(r => r.p_value))
			const sh = U5.gboost.shrinkage, inc = U5.bma.inclusion
			const items = [
				`Averaging trees works: one tree ${tree.auc_fold_mean.toFixed(3)} → random forest ${rf.auc_fold_mean.toFixed(3)} CV AUC, because variance falls from ${bv['CART (full depth)'].variance.toFixed(3)} to ${bv['Random Forest (300)'].variance.toFixed(3)}.`,
				...(minp >= 0.05 ? [
					`No ensemble beats logistic regression (${lr.auc_fold_mean.toFixed(3)}) by a significant margin: the smallest corrected p-value is ${minp.toFixed(2)}. The best, ${best.model.replace(' (scratch)', '')}, reaches ${best.auc_fold_mean.toFixed(3)}.`,
					`So the at-risk signal here is close to additive in the survey answers. Ensembles pay off for high-variance learners, not for an already well-specified model.`,
				] : [`${sig.find(r => r.p_value === minp).model} beats logistic regression significantly (p = ${minp.toFixed(3)}); best CV AUC ${best.auc_fold_mean.toFixed(3)}.`]),
				`BMA is sure about grades and attendance (inclusion ${inc.grades.toFixed(2)}, ${inc.attendance.toFixed(2)}) and drops study habits (${inc['study habits'].toFixed(2)}).`,
				`Shrinkage matters: with ν = ${sh[0].learning_rate} boosting over-fits after ${sh[0].best_iter} trees; ν = ${sh.at(-1).learning_rate} gives the lowest test log-loss (${sh.at(-1).best_test_logloss.toFixed(3)}).`,
			]
			ul.innerHTML = items.map(t => `<li>${t}</li>`).join('')
			fade(ul.querySelectorAll('li'), { stagger: 0.12, delay: 0.4 })
		},
	}

	/* ------------------------------------------------------------ one-off text fills */
	;(function fills() {
		const set = (id, v) => { const e = document.getElementById(id); if (e) e.textContent = v }
		const bvr = Object.fromEntries(U5.bias_variance.rows.map(r => [r.model, r])), fs = U5.adaboost.first_stump
		set('nModels', U5.compare_cv.length)
		set('aucTree', cvRow('CART (scratch').auc_fold_mean.toFixed(3)); set('aucPruned', cvRow('CART (pruned').auc_fold_mean.toFixed(3)); set('aucLR', cvRow('Logistic').auc_fold_mean.toFixed(3))
		set('stump1', `${short(fs.feature)} ≤ ${fs.threshold} · ε = ${fs.eps.toFixed(3)} · α = ${fs.alpha.toFixed(2)}`)
		set('varTree', bvr['CART (full depth)'].variance.toFixed(3)); set('varRF', bvr['Random Forest (300)'].variance.toFixed(3))
		set('bestName', U5.best_ensemble.replace(' (scratch)', ''))
		const st = document.getElementById('shrinkTable')
		if (st) table(st, ['ν', 'best m', 'test log-loss'], U5.gboost.shrinkage.map(g => [g.learning_rate, g.best_iter, g.best_test_logloss.toFixed(3)]), { best: U5.gboost.shrinkage.length - 1 })
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
		return
	}
	const start = +(location.hash.match(/^#s(\d+)$/)?.[1] || 1) - 1
	const go = () => show(start, 1)
	document.fonts?.ready ? document.fonts.ready.then(go) : go()
})()
