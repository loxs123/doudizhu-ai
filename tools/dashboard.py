#!/usr/bin/env python3
"""
斗地主训练 · 离线网页看板 (零依赖, 纯 Python 标准库)

与训练代码完全解耦: 只读取训练写出的 logs/metrics.jsonl, 不 import 任何训练模块。
图表用原生 canvas 绘制, 不依赖任何 CDN / 第三方 JS, 断网也能用。

用法:
    python tools/dashboard.py                       # 读 logs/metrics.jsonl, 监听 8000
    python tools/dashboard.py --metrics logs/metrics.jsonl --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_metrics(path):
    rows = []
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # 训练正写到一半的半行, 忽略
    return rows


PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>斗地主训练看板</title>
<style>
  :root{
    --bg:#0e1320; --panel:#161d2e; --panel2:#1d2740; --line:#2a3550;
    --txt:#e7ecf5; --muted:#8294b3; --accent:#4f9cff;
    --c0:#4f9cff; --c1:#ff8a5b; --c2:#5be3a0; --good:#41d68a; --bad:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--txt);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;}
  header{display:flex;align-items:center;gap:14px;padding:16px 22px;
    border-bottom:1px solid var(--line);background:var(--panel);}
  header h1{font-size:17px;margin:0;font-weight:600;letter-spacing:.3px}
  .status{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
  .dot{width:9px;height:9px;border-radius:50%;background:var(--bad);box-shadow:0 0 8px var(--bad)}
  .dot.live{background:var(--good);box-shadow:0 0 8px var(--good)}
  .wrap{padding:18px 22px;max-width:1280px;margin:0 auto}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:18px}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .card .k{font-size:12px;color:var(--muted);margin-bottom:6px}
  .card .v{font-size:26px;font-weight:700}
  .card .v small{font-size:14px;color:var(--muted);font-weight:500}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:880px){.grid{grid-template-columns:1fr}}
  .chart{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
  .chart h3{margin:0 0 4px;font-size:14px;font-weight:600}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-bottom:6px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:12px;height:3px;border-radius:2px;display:inline-block}
  canvas{width:100%;height:230px;display:block}
  table{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}
  th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}
  th{color:var(--muted);font-weight:500;text-align:right}
  th:first-child,td:first-child{text-align:left}
  tbody tr:hover{background:var(--panel2)}
  .tablecard{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin-top:16px}
  .empty{color:var(--muted);padding:30px;text-align:center}
</style>
</head>
<body>
<header>
  <h1>🃏 斗地主训练看板</h1>
  <div class="status">
    <span id="mode"></span>
    <span class="dot" id="dot"></span>
    <span id="stat">连接中…</span>
  </div>
</header>
<div class="wrap">
  <div class="cards" id="cards"></div>
  <div class="grid">
    <div class="chart">
      <h3>胜率</h3>
      <div class="legend">
        <span><i style="background:var(--c0)"></i>rollout(对当前对手)</span>
        <span><i style="background:var(--c2)"></i>eval(对随机农民)</span>
      </div>
      <canvas id="winrate"></canvas>
    </div>
    <div class="chart">
      <h3>Q Loss (各座位)</h3>
      <div class="legend" id="lg_loss"></div>
      <canvas id="loss"></canvas>
    </div>
    <div class="chart">
      <h3>Explained Variance (各座位)</h3>
      <div class="legend" id="lg_ev"></div>
      <canvas id="ev"></canvas>
    </div>
    <div class="chart">
      <h3>Q 预测均值 (各座位)</h3>
      <div class="legend" id="lg_q"></div>
      <canvas id="qmean"></canvas>
    </div>
  </div>
  <div class="tablecard">
    <h3 style="margin:0 0 6px;font-size:14px">最近 epoch</h3>
    <div id="tablebox"></div>
  </div>
</div>
<script>
const SEAT_COLORS = ['#4f9cff','#ff8a5b','#5be3a0'];
const SEAT_NAME = {0:'seat0 地主', 1:'seat1 农民', 2:'seat2 农民'};
let DATA = [];

function fmtPct(v){ return v==null||isNaN(v) ? '—' : (v*100).toFixed(1)+'%'; }

function seatIds(){
  const s = new Set();
  DATA.forEach(d => (d.seats||[]).forEach(x => s.add(x.id)));
  return [...s].sort();
}
function seatSeries(key){
  return seatIds().map(id => ({
    label: SEAT_NAME[id]||('seat'+id), color: SEAT_COLORS[id%3],
    points: DATA.filter(d => (d.seats||[]).some(x => x.id===id))
      .map(d => ({x:d.epoch, y:(d.seats.find(x=>x.id===id)||{})[key]}))
      .filter(p => p.y!=null && !isNaN(p.y))
  }));
}
function legend(elId, series){
  document.getElementById(elId).innerHTML = series.map(s =>
    `<span><i style="background:${s.color}"></i>${s.label}</span>`).join('');
}

function drawChart(canvasId, series, opts){
  opts = opts || {};
  const cv = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W*dpr; cv.height = H*dpr;
  const ctx = cv.getContext('2d'); ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const pad = {l:48, r:14, t:10, b:24};
  const pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;

  let xs=[], ys=[];
  series.forEach(s => s.points.forEach(p => {xs.push(p.x); ys.push(p.y);}));
  if(xs.length===0){
    ctx.fillStyle='#667a99'; ctx.font='13px system-ui';
    ctx.fillText('等待数据…', pad.l, pad.t+24); return;
  }
  let xmin=Math.min(...xs), xmax=Math.max(...xs);
  let ymin = opts.ymin!=null ? opts.ymin : Math.min(...ys);
  let ymax = opts.ymax!=null ? opts.ymax : Math.max(...ys);
  if(xmin===xmax) xmax=xmin+1;
  if(ymin===ymax){ ymin-=0.5; ymax+=0.5; }
  else if(opts.ymin==null && opts.ymax==null){ const m=(ymax-ymin)*0.08; ymin-=m; ymax+=m; }

  const sx = x => pad.l + (x-xmin)/(xmax-xmin)*pw;
  const sy = y => pad.t + (1-(y-ymin)/(ymax-ymin))*ph;

  // grid + y ticks
  ctx.font='11px system-ui'; ctx.textBaseline='middle';
  for(let i=0;i<=4;i++){
    const yy = ymin+(ymax-ymin)*i/4, py = sy(yy);
    ctx.strokeStyle='#222e47'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pad.l,py); ctx.lineTo(pad.l+pw,py); ctx.stroke();
    ctx.fillStyle='#7b8aa3'; ctx.textAlign='right';
    ctx.fillText(yy.toFixed(2), pad.l-6, py);
  }
  // x ticks
  ctx.textAlign='center'; ctx.textBaseline='top';
  const xticks = Math.min(6, xmax-xmin+1);
  for(let i=0;i<=xticks;i++){
    const xx = Math.round(xmin+(xmax-xmin)*i/xticks);
    ctx.fillStyle='#7b8aa3';
    ctx.fillText(xx, sx(xx), pad.t+ph+6);
  }
  // lines
  series.forEach(s => {
    if(s.points.length===0) return;
    ctx.strokeStyle=s.color; ctx.lineWidth=2; ctx.beginPath();
    s.points.forEach((p,i) => { const X=sx(p.x), Y=sy(p.y); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
    ctx.stroke();
    // last point dot
    const lp = s.points[s.points.length-1];
    ctx.fillStyle=s.color; ctx.beginPath(); ctx.arc(sx(lp.x),sy(lp.y),3,0,7); ctx.fill();
  });
}

function render(){
  const cardsEl = document.getElementById('cards');
  if(DATA.length===0){
    cardsEl.innerHTML = '<div class="empty" style="grid-column:1/-1">还没有指标数据。启动训练后会自动出现。</div>';
  } else {
    const last = DATA[DATA.length-1];
    const lastEval = [...DATA].reverse().find(d => d.eval_win!=null);
    const cards = [
      ['当前 epoch', last.epoch + ' <small>/ 共 '+(DATA.length)+' 条</small>'],
      ['地主胜率 (rollout)', fmtPct(last.rollout_win)],
      ['地主胜率 (vs 随机)', lastEval ? fmtPct(lastEval.eval_win) : '—'],
      ['本 epoch 用时', (last.elapsed!=null? last.elapsed.toFixed(1):'—') + ' <small>s</small>'],
    ];
    cardsEl.innerHTML = cards.map(c => `<div class="card"><div class="k">${c[0]}</div><div class="v">${c[1]}</div></div>`).join('');
  }

  drawChart('winrate', [
    {label:'rollout', color:'#4f9cff', points: DATA.filter(d=>d.rollout_win!=null).map(d=>({x:d.epoch,y:d.rollout_win}))},
    {label:'eval', color:'#5be3a0', points: DATA.filter(d=>d.eval_win!=null).map(d=>({x:d.epoch,y:d.eval_win}))},
  ], {ymin:0, ymax:1});

  const loss = seatSeries('q_loss'); legend('lg_loss', loss); drawChart('loss', loss, {ymin:0});
  const ev = seatSeries('explained_var'); legend('lg_ev', ev); drawChart('ev', ev, {ymax:1});
  const qm = seatSeries('q_mean'); legend('lg_q', qm); drawChart('qmean', qm);

  // table (last 12)
  const rows = DATA.slice(-12).reverse();
  const tb = document.getElementById('tablebox');
  if(rows.length===0){ tb.innerHTML='<div class="empty">—</div>'; }
  else {
    let html = '<table><thead><tr><th>epoch</th><th>rollout</th><th>eval</th>'+
      seatIds().map(id=>`<th>${SEAT_NAME[id]||'s'+id} Qloss</th>`).join('')+'<th>用时</th></tr></thead><tbody>';
    rows.forEach(d => {
      html += `<tr><td>${d.epoch}</td><td>${fmtPct(d.rollout_win)}</td><td>${fmtPct(d.eval_win)}</td>`+
        seatIds().map(id=>{const s=(d.seats||[]).find(x=>x.id===id); return `<td>${s?s.q_loss.toFixed(3):'—'}</td>`;}).join('')+
        `<td>${d.elapsed!=null?d.elapsed.toFixed(1)+'s':'—'}</td></tr>`;
    });
    tb.innerHTML = html + '</tbody></table>';
  }
}

async function poll(){
  const dot = document.getElementById('dot'), stat = document.getElementById('stat'), modeEl=document.getElementById('mode');
  try{
    const r = await fetch('/api/metrics', {cache:'no-store'});
    DATA = await r.json();
    dot.classList.add('live');
    const now = new Date().toLocaleTimeString();
    stat.textContent = `已更新 ${now} · ${DATA.length} epoch`;
    if(DATA.length) modeEl.textContent = '模式: ' + (DATA[DATA.length-1].mode||'');
    render();
  }catch(e){
    dot.classList.remove('live');
    stat.textContent = '连接断开 (训练/看板已停止?)';
  }
}
window.addEventListener('resize', render);
poll();
setInterval(poll, 3000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    metrics_path = None

    def log_message(self, *a):
        pass  # 静音访问日志

    def _send(self, body, ctype):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith('/api/metrics'):
            self._send(json.dumps(load_metrics(self.metrics_path)), 'application/json; charset=utf-8')
        elif self.path == '/' or self.path.startswith('/index'):
            self._send(PAGE, 'text/html; charset=utf-8')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="斗地主训练离线看板")
    ap.add_argument('--metrics', default='logs/metrics.jsonl', help='训练写出的指标文件')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    Handler.metrics_path = args.metrics
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板已启动:  http://{args.host}:{args.port}")
    print(f"读取指标:    {os.path.abspath(args.metrics)}")
    print("按 Ctrl+C 退出。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == '__main__':
    main()
