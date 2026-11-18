# usage: python3 source/render.py source/nibtera-online.html posts/nibtera-online.png
import sys
from playwright.sync_api import sync_playwright
src,out=sys.argv[1],sys.argv[2]
import os
with sync_playwright() as p:
    b=p.chromium.launch();pg=b.new_page(viewport={"width":1254,"height":1254})
    pg.goto("file://"+os.path.abspath(src));pg.wait_for_timeout(1000);pg.screenshot(path=out);b.close()
