# -*- coding: utf-8 -*-
"""一键重拍 SPXH 工作台全部页签截图。

用法：
    python -m spxh serve            # 另开一个终端启动工作台
    python "docs/发布/工具-截图.py"  # 可选: --base http://127.0.0.1:8760/ --out 目录

依赖: pip install playwright  (会自动使用本机 Chrome)
"""
import argparse, os, time, urllib.parse
from playwright.sync_api import sync_playwright

QPSK = "data/demo/qpsk_snr+20.0_000.sigmf-meta"
CONV = "data/demo/qpsk_conv_snr+10.0_000.sigmf-meta"
FFSK = "data/demo/2fsk_snr+5.0_000.sigmf-meta"

# (输出名, 案例, 页签, 动作)
JOBS = [
    ("01-频谱与瀑布.png", FFSK, "spectrum", None),
    ("02-参数估计.png",   QPSK, "estimate", None),
    ("03-特征.png",       QPSK, "features", None),
    ("04-识别.png",       QPSK, "classify", None),
    ("05-解调.png",       QPSK, "demod",    None),
    ("06-译码.png",       QPSK, "fec",      None),
    ("07-帧.png",         CONV, "frame",    None),
    ("08-谱相关.png",     QPSK, "ssca",     None),
    ("09-报告-离线确定性.png", CONV, "narrate", "narrate_offline"),
    ("10-报告-LLM模式.png",    CONV, "narrate", "narrate_auto"),
    ("11-代理-LLM时间线.png",  CONV, "agent",   "agent_auto"),
    ("12-代理-离线回退.png",   CONV, "agent",   "agent_offline"),
]

READY = """(tab) => {
  const p = document.querySelector('#panel-' + tab);
  if (!p) return 'nopanel';
  const t = p.innerText || '';
  if (t.indexOf('加载中') >= 0 || t.indexOf('正在加载') >= 0) return 'loading';
  if (tab === 'agent') {
    const st = (document.querySelector('#agent-status') || {}).innerText || '';
    const rd = (document.querySelector('#agent-rounds') || {}).innerText || '';
    if (st.indexOf('运行中') >= 0) return 'running';
    if (rd.trim().length < 40) return 'empty';
  }
  if (tab === 'narrate') {
    const sec = (document.querySelector('#narrate-sections') || {}).innerText || '';
    if (sec.trim().length < 20) return 'empty';
  }
  return 'ok';
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8760/")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "配图"))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--only", default="", help="只拍名字包含该子串的页签，例如 --only 报告")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--hide-scrollbars"])
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080},
                                  device_scale_factor=1, locale="zh-CN")
        page = ctx.new_page()
        for name, path, tab, action in JOBS:
            if a.only and a.only not in name:
                continue
            url = a.base + "?" + urllib.parse.urlencode({"path": path, "tab": tab})
            page.goto(url, wait_until="load", timeout=60000)
            page.wait_for_timeout(2500)
            if action == "narrate_offline":
                page.select_option("#ctl-narrate-provider", "offline"); page.click("#ctl-narrate-reload")
            elif action == "narrate_auto":
                page.select_option("#ctl-narrate-provider", "auto"); page.click("#ctl-narrate-reload")
            elif action == "agent_offline":
                page.select_option("#ctl-agent-provider", "offline"); page.click("#agent-run")
            elif action == "agent_auto":
                page.select_option("#ctl-agent-provider", "auto"); page.click("#agent-run")
            t0 = time.time()
            while time.time() - t0 < a.timeout:
                if page.evaluate(READY, tab) == "ok":
                    break
                page.wait_for_timeout(1500)
            page.wait_for_timeout(2000)
            page.screenshot(path=os.path.join(a.out, name), full_page=False)
            print("saved", name, round(time.time() - t0, 1), "s")
        ctx.close(); browser.close()


if __name__ == "__main__":
    main()
