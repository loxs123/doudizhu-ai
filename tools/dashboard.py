#!/usr/bin/env python3
"""
斗地主训练 · 离线网页看板 (卡通风, 零依赖, 纯 Python 标准库)

与训练代码完全解耦: 只读取训练写出的 logs/metrics.jsonl 与 logs/eval_losses.json,
不 import 任何训练模块。图表用原生 canvas 绘制, 不依赖任何 CDN / 第三方 JS, 断网也能用。

用法:
    python tools/dashboard.py
    python tools/dashboard.py --metrics logs/metrics.jsonl --losses logs/eval_losses.json --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_jsonl(path):
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
                    pass
    return rows


def load_json(path):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>斗地主训练看板</title>
<style>
  :root{
    --bg:#fff7ec; --ink:#2b2b2b; --card:#ffffff; --line:#2b2b2b;
    --shadow:5px 5px 0 #2b2b2b; --shadow-sm:3px 3px 0 #2b2b2b;
    --lord:#ffcb3d; --farm:#7fdc8b; --deck:#9ad0ff;
    --c0:#4f9cff; --c1:#ff8a5b; --c2:#4cc38a;
    --red:#ff5d5d; --orange:#ff9f1c; --purple:#b388ff; --good:#21c47a; --bad:#ff5d5d;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:"Comic Sans MS","Chalkboard SE",-apple-system,"PingFang SC","Microsoft YaHei",system-ui,sans-serif;
    background-image:radial-gradient(#00000010 1.4px, transparent 1.4px);background-size:22px 22px;}
  header{display:flex;align-items:center;gap:14px;padding:16px 22px;margin:18px 22px 0;
    background:var(--lord);border:3px solid var(--line);border-radius:18px;box-shadow:var(--shadow);}
  header h1{font-size:20px;margin:0;font-weight:800;letter-spacing:.5px}
  .status{margin-left:auto;display:flex;align-items:center;gap:8px;font-size:13px;font-weight:700}
  .dot{width:12px;height:12px;border-radius:50%;background:var(--bad);border:2px solid var(--line)}
  .dot.live{background:var(--good)}
  .wrap{padding:18px 22px 60px;max-width:1280px;margin:0 auto}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px;margin-bottom:20px}
  .card{background:var(--card);border:3px solid var(--line);border-radius:18px;padding:14px 16px;box-shadow:var(--shadow-sm)}
  .card .k{font-size:13px;font-weight:700;margin-bottom:6px;opacity:.8}
  .card .v{font-size:28px;font-weight:800}
  .card .v small{font-size:14px;font-weight:700;opacity:.6}
  .card.gold{background:#fff1c2}.card.green{background:#e3f8e6}.card.blue{background:#e3f1ff}.card.pink{background:#ffe6ec}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
  @media(max-width:880px){.grid{grid-template-columns:1fr}}
  .chart,.tablecard,.losscard{background:var(--card);border:3px solid var(--line);border-radius:18px;
    padding:14px 18px;box-shadow:var(--shadow-sm);margin-top:18px}
  .chart{margin-top:0}
  h3{margin:0 0 8px;font-size:16px;font-weight:800}
  .legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;font-weight:700;margin-bottom:6px}
  .legend span{display:inline-flex;align-items:center;gap:6px}
  .legend i{width:14px;height:5px;border-radius:3px;display:inline-block;border:1px solid var(--line)}
  canvas{width:100%;height:230px;display:block}
  .empty{padding:30px;text-align:center;font-weight:700;opacity:.6}

  /* 卡片牌 chip */
  .chip{display:inline-block;background:#fff;border:2px solid var(--line);border-radius:8px;
    padding:1px 7px;margin:2px 3px;font-weight:800;font-size:13px;box-shadow:2px 2px 0 #2b2b2b}
  .chip.two{background:#ffe08a}.chip.joker{background:#ff9b9b}
  .chip.pass{background:#e8e8e8;opacity:.8}.chip.empty{background:#e8e8e8;opacity:.6}

  /* 败局回放 */
  .loss-meta{font-weight:800;margin-bottom:12px;font-size:14px}
  .game{border:3px dashed var(--line);border-radius:16px;padding:12px 14px;margin-bottom:16px;background:#fffdf6}
  .game-head{font-weight:800;font-size:15px;margin-bottom:10px;background:var(--red);color:#fff;
    display:inline-block;padding:4px 12px;border:2px solid var(--line);border-radius:12px;box-shadow:2px 2px 0 #2b2b2b}
  .hands{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
  .hand{padding:6px 10px;border:2px solid var(--line);border-radius:12px}
  .hand b{margin-right:8px}
  .hand.lord{background:#fff1c2}.hand.farm{background:#e3f8e6}.hand.deck{background:#e3f1ff}
  .moves{display:flex;flex-direction:column;gap:7px}
  .move{border:2px solid var(--line);border-radius:12px;padding:6px 9px;background:#fff}
  .move-top{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .mv-no{font-weight:800;background:var(--ink);color:#fff;border-radius:8px;padding:1px 8px;font-size:12px}
  .mv-who{font-weight:800;border:2px solid var(--line);border-radius:8px;padding:1px 8px;font-size:12px}
  .move.p0 .mv-who{background:var(--lord)}.move.p1 .mv-who,.move.p2 .mv-who{background:var(--farm)}
  .mv-remain{margin-top:6px;display:flex;flex-direction:column;gap:3px}
  .rem{font-size:12px;opacity:.85;line-height:1.5}
  .rem.active{opacity:1;font-weight:800;background:#fff7d6;border-radius:8px;padding:1px 4px}
  .rem .cnt{display:inline-block;min-width:34px;font-weight:800}

  /* 全字段表 */
  .scrollx{overflow-x:auto}
  table{border-collapse:separate;border-spacing:0;font-size:12.5px;white-space:nowrap}
  th,td{padding:7px 10px;text-align:right;border-bottom:2px solid #eee}
  th{font-weight:800;text-align:right;position:sticky;top:0;background:#fff}
  th:first-child,td:first-child{text-align:left}
  tbody tr:nth-child(even){background:#fffaf0}
  tbody tr:hover{background:#fff1c2}
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
      <h3>🏆 胜率</h3>
      <div class="legend">
        <span><i style="background:var(--c0)"></i>rollout(当前对手)</span>
        <span><i style="background:var(--c2)"></i>eval(随机农民)</span>
        <span><i style="background:var(--purple)"></i>best</span>
      </div>
      <canvas id="winrate"></canvas>
    </div>
    <div class="chart">
      <h3>📉 Q Loss (各座位)</h3>
      <div class="legend" id="lg_loss"></div>
      <canvas id="loss"></canvas>
    </div>
    <div class="chart">
      <h3>📊 Explained Variance (各座位)</h3>
      <div class="legend" id="lg_ev"></div>
      <canvas id="ev"></canvas>
    </div>
    <div class="chart">
      <h3>🎲 策略熵 (探索多样性, 0=塌缩 · 1=均匀)</h3>
      <div class="legend"><span><i style="background:var(--purple)"></i>softmax(Q/τ) 归一化熵</span></div>
      <canvas id="entropy"></canvas>
    </div>
  </div>

  <div class="losscard">
    <h3>🎴 评估败局回放 (地主 vs 随机农民, 每手含三家剩余牌)</h3>
    <div id="losses"><div class="empty">等待评估数据…</div></div>
  </div>

  <div class="tablecard">
    <h3>📋 各轮完整指标</h3>
    <div class="scrollx" id="tablebox"></div>
  </div>
</div>
<script>
const SEAT_COLORS = ['#4f9cff','#ff8a5b','#4cc38a'];
const SEAT_NAME = {0:'seat0 地主', 1:'seat1 农民', 2:'seat2 农民'};
let DATA = [];
let LOSSES = null;

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
  if(xs.length===0){ ctx.fillStyle='#999'; ctx.font='13px system-ui'; ctx.fillText('等待数据…', pad.l, pad.t+24); return; }
  let xmin=Math.min(...xs), xmax=Math.max(...xs);
  let ymin = opts.ymin!=null ? opts.ymin : Math.min(...ys);
  let ymax = opts.ymax!=null ? opts.ymax : Math.max(...ys);
  if(xmin===xmax) xmax=xmin+1;
  if(ymin===ymax){ ymin-=0.5; ymax+=0.5; }
  else if(opts.ymin==null && opts.ymax==null){ const m=(ymax-ymin)*0.08; ymin-=m; ymax+=m; }
  const sx = x => pad.l + (x-xmin)/(xmax-xmin)*pw;
  const sy = y => pad.t + (1-(y-ymin)/(ymax-ymin))*ph;
  ctx.font='11px system-ui'; ctx.textBaseline='middle';
  for(let i=0;i<=4;i++){
    const yy = ymin+(ymax-ymin)*i/4, py = sy(yy);
    ctx.strokeStyle='#eee'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(pad.l,py); ctx.lineTo(pad.l+pw,py); ctx.stroke();
    ctx.fillStyle='#888'; ctx.textAlign='right'; ctx.fillText(yy.toFixed(2), pad.l-6, py);
  }
  ctx.textAlign='center'; ctx.textBaseline='top';
  const xticks = Math.min(6, xmax-xmin+1);
  for(let i=0;i<=xticks;i++){
    const xx = Math.round(xmin+(xmax-xmin)*i/xticks);
    ctx.fillStyle='#888'; ctx.fillText(xx, sx(xx), pad.t+ph+6);
  }
  series.forEach(s => {
    if(s.points.length===0) return;
    ctx.strokeStyle=s.color; ctx.lineWidth=3; ctx.lineJoin='round'; ctx.beginPath();
    s.points.forEach((p,i) => { const X=sx(p.x), Y=sy(p.y); i?ctx.lineTo(X,Y):ctx.moveTo(X,Y); });
    ctx.stroke();
    const lp = s.points[s.points.length-1];
    ctx.fillStyle=s.color; ctx.strokeStyle='#2b2b2b'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.arc(sx(lp.x),sy(lp.y),4,0,7); ctx.fill(); ctx.stroke();
  });
}

function chips(str){
  if(str==null || str==='') return '<span class="chip empty">空</span>';
  if(str==='过') return '<span class="chip pass">过</span>';
  return str.split(' ').map(t=>{
    let cls='chip';
    if(t==='大王'||t==='小王') cls+=' joker';
    else if(t==='2') cls+=' two';
    return `<span class="${cls}">${t}</span>`;
  }).join('');
}
function cnt(str){ return (str==null||str===''||str==='过') ? 0 : str.split(' ').length; }

function render(){
  const cardsEl = document.getElementById('cards');
  if(DATA.length===0){
    cardsEl.innerHTML = '<div class="empty" style="grid-column:1/-1">还没有指标数据，启动训练后会自动出现 🚀</div>';
  } else {
    const last = DATA[DATA.length-1];
    const lastEval = [...DATA].reverse().find(d => d.eval_win!=null);
    const best = [...DATA].reverse().find(d => d.best_eval!=null);
    const cards = [
      ['gold','📅 当前 epoch', last.epoch + ' <small>/ '+DATA.length+' 条</small>'],
      ['blue','🤖 地主胜率 (rollout)', fmtPct(last.rollout_win)],
      ['green','🎯 地主胜率 (vs 随机)', lastEval ? fmtPct(lastEval.eval_win) : '—'],
      ['pink','🏆 历史最佳', best ? fmtPct(best.best_eval) : '—'],
      ['','🎲 策略熵', last.policy_entropy!=null ? last.policy_entropy.toFixed(3) : '—'],
      ['','⏱ 本轮用时', (last.elapsed!=null? last.elapsed.toFixed(1):'—') + ' <small>s</small>'],
    ];
    cardsEl.innerHTML = cards.map(c => `<div class="card ${c[0]}"><div class="k">${c[1]}</div><div class="v">${c[2]}</div></div>`).join('');
  }

  drawChart('winrate', [
    {label:'rollout', color:'#4f9cff', points: DATA.filter(d=>d.rollout_win!=null).map(d=>({x:d.epoch,y:d.rollout_win}))},
    {label:'eval', color:'#4cc38a', points: DATA.filter(d=>d.eval_win!=null).map(d=>({x:d.epoch,y:d.eval_win}))},
    {label:'best', color:'#b388ff', points: DATA.filter(d=>d.best_eval!=null).map(d=>({x:d.epoch,y:d.best_eval}))},
  ], {ymin:0, ymax:1});
  const loss = seatSeries('q_loss'); legend('lg_loss', loss); drawChart('loss', loss, {ymin:0});
  const ev = seatSeries('explained_var'); legend('lg_ev', ev); drawChart('ev', ev, {ymax:1});
  drawChart('entropy', [{label:'策略熵', color:'#b388ff',
    points: DATA.filter(d=>d.policy_entropy!=null).map(d=>({x:d.epoch,y:d.policy_entropy}))}], {ymin:0, ymax:1});

  renderLosses();
  renderTable();
}

function renderLosses(){
  const box = document.getElementById('losses');
  if(!LOSSES || !LOSSES.games || LOSSES.games.length===0){
    box.innerHTML = '<div class="empty">暂无评估败局数据 (下次 eval 后出现) 🎉</div>'; return;
  }
  let html = `<div class="loss-meta">第 ${LOSSES.epoch} 轮评估 · 胜率 ${fmtPct(LOSSES.eval_win)} · 败局共 ${LOSSES.n_losses_total} 局 (展示前 ${LOSSES.games.length} 局)</div>`;
  LOSSES.games.forEach((g, gi) => {
    html += `<div class="game">`;
    html += `<div class="game-head">败局 #${gi+1} · 共 ${g.n_moves} 手 · 农民 P${g.winner} 先跑 🏃💨</div>`;
    html += `<div class="hands">`;
    html += `<div class="hand lord"><b>👑 地主P0 [${cnt(g.init[0])}]</b>${chips(g.init[0])}</div>`;
    html += `<div class="hand farm"><b>🧑‍🌾 农民P1 [${cnt(g.init[1])}]</b>${chips(g.init[1])}</div>`;
    html += `<div class="hand farm"><b>🧑‍🌾 农民P2 [${cnt(g.init[2])}]</b>${chips(g.init[2])}</div>`;
    html += `<div class="hand deck"><b>🂠 底牌</b>${chips(g.end)}</div>`;
    html += `</div><div class="moves">`;
    g.moves.forEach((m, i) => {
      html += `<div class="move p${m.player}"><div class="move-top">`;
      html += `<span class="mv-no">${i+1}</span><span class="mv-who">P${m.player}</span>`;
      html += `<span class="mv-play">出 ${chips(m.play)}</span></div>`;
      html += `<div class="mv-remain">`;
      const tag = ['👑P0','🧑‍🌾P1','🧑‍🌾P2'];
      for(let s=0;s<3;s++){
        html += `<div class="rem ${s===m.player?'active':''}"><span class="cnt">${tag[s]} ${cnt(m.remain[s])}张</span> ${chips(m.remain[s])}</div>`;
      }
      html += `</div></div>`;
    });
    html += `</div></div>`;
  });
  box.innerHTML = html;
}

function renderTable(){
  const box = document.getElementById('tablebox');
  if(DATA.length===0){ box.innerHTML='<div class="empty">—</div>'; return; }
  const ids = seatIds();
  const top = [['epoch','epoch'],['rollout_win','rollout'],['eval_win','eval'],['best_eval','best'],
               ['policy_entropy','熵'],['q_gap','Q差'],['elapsed','用时s'],['mode','模式']];
  const seatKeys = [['q_loss','Qloss'],['q_mean','Qmean'],['q_std','Qstd'],
                    ['explained_var','EV'],['grad_norm','grad'],['return_mean','ret']];
  let head = '<tr>';
  top.forEach(c => head += `<th>${c[1]}</th>`);
  ids.forEach(id => seatKeys.forEach(k => head += `<th>s${id}·${k[1]}</th>`));
  head += '</tr>';
  const fmtCell = (k,v) => {
    if(v==null) return '—';
    if(k==='rollout_win'||k==='eval_win'||k==='best_eval') return (v*100).toFixed(1)+'%';
    if(typeof v==='number') return Number.isInteger(v)? v : v.toFixed(3);
    return v;
  };
  let body = '';
  [...DATA].reverse().forEach(d => {
    let row = '<tr>';
    top.forEach(c => row += `<td>${fmtCell(c[0], d[c[0]])}</td>`);
    ids.forEach(id => {
      const s = (d.seats||[]).find(x=>x.id===id) || {};
      seatKeys.forEach(k => row += `<td>${fmtCell(k[0], s[k[0]])}</td>`);
    });
    row += '</tr>'; body += row;
  });
  box.innerHTML = `<table><thead>${head}</thead><tbody>${body}</tbody></table>`;
}

async function poll(){
  const dot = document.getElementById('dot'), stat = document.getElementById('stat'), modeEl=document.getElementById('mode');
  try{
    const [m, l] = await Promise.all([
      fetch('/api/metrics', {cache:'no-store'}).then(r=>r.json()),
      fetch('/api/losses', {cache:'no-store'}).then(r=>r.json()).catch(()=>null),
    ]);
    DATA = m; LOSSES = l;
    dot.classList.add('live');
    stat.textContent = `已更新 ${new Date().toLocaleTimeString()} · ${DATA.length} epoch`;
    if(DATA.length) modeEl.textContent = '模式: ' + (DATA[DATA.length-1].mode||'');
    render();
  }catch(e){
    dot.classList.remove('live');
    stat.textContent = '连接断开 🔌';
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
    losses_path = None

    def log_message(self, *a):
        pass

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
            self._send(json.dumps(load_jsonl(self.metrics_path)), 'application/json; charset=utf-8')
        elif self.path.startswith('/api/losses'):
            self._send(json.dumps(load_json(self.losses_path)), 'application/json; charset=utf-8')
        elif self.path == '/' or self.path.startswith('/index'):
            self._send(PAGE, 'text/html; charset=utf-8')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    ap = argparse.ArgumentParser(description="斗地主训练离线看板 (卡通风)")
    ap.add_argument('--metrics', default='logs/metrics.jsonl', help='训练写出的指标文件')
    ap.add_argument('--losses', default='logs/eval_losses.json', help='评估败局回放文件')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8000)
    args = ap.parse_args()

    Handler.metrics_path = args.metrics
    Handler.losses_path = args.losses
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"看板已启动:  http://{args.host}:{args.port}")
    print(f"指标:        {os.path.abspath(args.metrics)}")
    print(f"败局回放:    {os.path.abspath(args.losses)}")
    print("按 Ctrl+C 退出。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == '__main__':
    main()
