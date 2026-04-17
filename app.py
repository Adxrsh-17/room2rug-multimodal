from __future__ import annotations
import base64, sys, time
from io import BytesIO
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title='Rug Search AI',
    page_icon='x1F9F6',
    layout='wide',
    initial_sidebar_state='expanded',
)

_CSS = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:#0f0f1a;color:#e2e8f0;}
.hero{background:linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);
      border-radius:20px;padding:2.2rem 2.8rem 1.8rem;margin-bottom:1.5rem;
      box-shadow:0 16px 48px rgba(102,126,234,.35);}
.hero h1{font-size:2.4rem;font-weight:800;color:#fff;margin:0 0 .4rem;}
.hero p{font-size:1rem;color:rgba(255,255,255,.88);margin:0;}
.rug-card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
          border-radius:16px;overflow:hidden;margin-bottom:1rem;
          transition:transform .18s,box-shadow .18s;position:relative;}
.rug-card:hover{transform:translateY(-4px);box-shadow:0 12px 36px rgba(118,75,162,.28);}
.rimgw{width:100%;height:200px;overflow:hidden;background:#1a1a2e;
       display:flex;align-items:center;justify-content:center;}
.rimgw img{width:100%;height:200px;object-fit:cover;display:block;transition:transform .3s;}
.rimgw img:hover{transform:scale(1.06);}
.rph{width:100%;height:200px;display:flex;align-items:center;justify-content:center;
     font-size:2.5rem;color:#4b5563;}
.rbadge{position:absolute;top:10px;left:10px;
        background:linear-gradient(135deg,#667eea,#764ba2);
        color:#fff;font-weight:800;font-size:.75rem;border-radius:10px;
        padding:3px 9px;box-shadow:0 2px 8px rgba(102,126,234,.55);z-index:10;}
.cbody{padding:.9rem 1rem 1rem;}
.ctitle{font-size:.9rem;font-weight:700;color:#f0f4ff;margin-bottom:.25rem;line-height:1.35;}
.cprice{font-size:1rem;font-weight:700;color:#34d399;margin-bottom:.6rem;}
.pills{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:.5rem;}
.pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:.71rem;font-weight:700;}
.pf{background:#312e81;color:#a5b4fc;border:1px solid #4338ca;}
.pi{background:#1e3a5f;color:#93c5fd;border:1px solid #2563eb;}
.pt{background:#14532d;color:#86efac;border:1px solid #16a34a;}
.pm{background:#451a03;color:#fcd34d;border:1px solid #d97706;}
.ps{background:#4a044e;color:#f0abfc;border:1px solid #a21caf;}
.blbl{font-size:.65rem;color:#6b7280;margin-top:4px;}
.bwrap{background:rgba(255,255,255,.08);border-radius:999px;height:5px;margin:3px 0 2px;}
.bfill{border-radius:999px;height:5px;}
.lconf{color:#f87171;font-size:.7rem;font-weight:700;margin-top:4px;}
.expl{font-size:.72rem;color:#9ca3af;font-style:italic;margin-top:6px;padding:6px 8px;
      background:rgba(255,255,255,.03);border-left:2px solid #764ba2;border-radius:4px;line-height:1.4;}
section[data-testid="stSidebar"]{background:#0d0d1a !important;}
.scard{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);
       border-radius:12px;padding:.75rem .9rem;margin-bottom:.75rem;font-size:.82rem;color:#9ca3af;}
.scard strong{color:#e2e8f0;}
.bok{background:linear-gradient(90deg,rgba(16,185,129,.15),rgba(16,185,129,.05));
     border:1px solid rgba(16,185,129,.3);border-radius:10px;padding:.6rem 1rem;
     color:#6ee7b7;font-weight:600;font-size:.88rem;margin-bottom:1rem;}
.bwarn{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3);
       border-radius:10px;padding:.6rem 1rem;color:#fcd34d;
       font-weight:600;font-size:.88rem;margin-bottom:1rem;}
div[data-testid="stMetric"] label{color:#9ca3af !important;font-size:.75rem !important;}
div[data-testid="stMetricValue"]{color:#e2e8f0 !important;font-size:1.1rem !important;font-weight:700 !important;}
div[data-testid="stTabs"] button{font-weight:700;font-size:.88rem;}
</style>
'''
st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False, ttl=3600)
def img_proxy(url: str) -> str:
    '''Download image server-side -> base64 data URI (bypasses tracking prevention).'''
    if not url:
        return ''
    try:
        import requests
        r = requests.get(url, timeout=8, headers={'User-Agent': 'Mozilla/5.0 (compatible)'})
        if r.status_code == 200:
            ct = r.headers.get('Content-Type', 'image/jpeg').split(';')[0].strip()
            b64 = base64.b64encode(r.content).decode()
            return f'data:{ct};base64,{b64}'
    except Exception:
        pass
    return ''


@st.cache_resource(show_spinner='Loading catalog...')
def load_catalog():
    from src.preprocess import build_catalog
    return build_catalog(ROOT / 'data' / 'products.csv')


@st.cache_resource(show_spinner='Warming SentenceTransformer...')
def warm_st():
    from src.ranker import _get_model
    return _get_model()


def _sc(s):
    return '#10b981' if s >= 0.45 else ('#f59e0b' if s >= 0.28 else '#ef4444')


def _bar(s, lbl=''):
    pct = min(int(s * 100), 100)
    return (f'<div class="blbl">{lbl}</div>'
            f'<div class="bwrap"><div class="bfill" style="width:{pct}%;background:{_sc(s)};"></div></div>')


def _img(url, alt):
    src = img_proxy(url) if url else ''
    return (f'<img src="{src}" alt="{alt}">' if src
            else '<div class="rph">&#x1F9F6;</div>')


def _c1(r):
    rk = r.get('rank', '?')
    tt = (r.get('title') or 'Untitled')[:70]
    pr = r.get('price'); url = r.get('image_url') or ''
    fs = r.get('final_score', 0.); ms = r.get('metadata_score', 0.); ss = r.get('semantic_score', 0.)
    p = f'${pr:,.2f}' if pr else '-'
    return (
        f'<div class="rug-card"><div class="rbadge">#{rk}</div>'
        f'<div class="rimgw">{_img(url, tt)}</div>'
        f'<div class="cbody"><div class="ctitle">{tt}</div>'
        f'<div class="cprice">{p}</div>'
        f'<div class="pills">'
        f'<span class="pill pf">Score {fs:.3f}</span>'
        f'<span class="pill pm">Meta {ms:.3f}</span>'
        f'<span class="pill ps">Sem {ss:.3f}</span></div>'
        f'{_bar(fs, "Final score")}</div></div>'
    )


def _c2(r):
    rk = r.get('rank', '?')
    tt = (r.get('title') or 'Untitled')[:70]
    pr = r.get('price'); url = r.get('image_url') or ''
    fus = r.get('fusion_score', 0.); ims = r.get('image_similarity', 0.); txs = r.get('text_similarity', 0.)
    ex = r.get('explanation', ''); lc = r.get('low_confidence', False)
    p = f'${pr:,.2f}' if pr else '-'
    lch = '<div class="lconf">&#x26A0; Low confidence</div>' if lc else ''
    exh = f'<div class="expl">&#128172; {ex}</div>' if ex else ''
    return (
        f'<div class="rug-card"><div class="rbadge">#{rk}</div>'
        f'<div class="rimgw">{_img(url, tt)}</div>'
        f'<div class="cbody"><div class="ctitle">{tt}</div>'
        f'<div class="cprice">{p}</div>'
        f'<div class="pills">'
        f'<span class="pill pf">Fusion {fus:.3f}</span>'
        f'<span class="pill pi">Img {ims:.3f}</span>'
        f'<span class="pill pt">Txt {txs:.3f}</span></div>'
        f'{_bar(fus, "Fusion score")}{lch}{exh}</div></div>'
    )


def render_grid(results, mode='part1', cols=3):
    for i in range(0, len(results), cols):
        sc = st.columns(cols)
        for ci, r in enumerate(results[i:i + cols]):
            html = _c1(r) if mode == 'part1' else _c2(r)
            with sc[ci]:
                st.markdown(html, unsafe_allow_html=True)


# SIDEBAR
with st.sidebar:
    st.markdown('## Settings')
    top_k = st.slider('Results (Top-K)', 3, 15, 6)
    st.markdown('---')
    catalog = load_catalog()
    st.markdown(
        f'<div class="scard"><strong>Catalog</strong><br/>'
        f'{len(catalog)} unique products<br/>'
        f'<span style="font-size:.75rem;color:#6b7280;">data/products.csv</span></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="scard"><strong>Score Formulas</strong><br/><br/>'
        '<strong>Part 1</strong><br/>0.5 x metadata + 0.5 x semantic<br/><br/>'
        '<strong>Part 2 Mode B</strong><br/>0.6 x image_sim + 0.4 x text_sim</div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="scard"><strong>Models</strong><br/>'
        'SentenceTransformers all-MiniLM-L6-v2<br/>'
        'CLIP ViT-B/32 openai/clip-vit-base-patch32</div>',
        unsafe_allow_html=True)
    st.markdown('---')
    st.caption('Multimodal Rug Search 2026')


# HERO
st.markdown(
    '<div class="hero"><h1>Multimodal Rug Search</h1>'
    '<p>Part 1 - Structured text | Part 2 - Room image + CLIP + FAISS</p></div>',
    unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(['Part 1 - Text Search', 'Part 2 - Multimodal', 'Catalog Explorer'])

# TAB 1
with tab1:
    st.markdown('### Structured Text Search')
    st.caption('Regex + rule-based parser => metadata filter => SentenceTransformer ranking')
    cq, cb = st.columns([5, 1])
    with cq:
        query1 = st.text_input('q', placeholder='e.g. 8x10 beige traditional rug | runner blue',
                               label_visibility='collapsed')
    with cb:
        go1 = st.button('Search', type='primary', use_container_width=True)
    ec = st.columns(4)
    exs = ['8x10 beige traditional rug', 'runner 2x10 blue rug',
           'small bohemian accent rug', '9x12 navy geometric']
    picked = None
    for i, ex in enumerate(exs):
        if ec[i].button(ex, key=f'e{i}', use_container_width=True):
            picked = ex
    fq = picked or query1
    if (go1 or picked) and fq:
        from src.query_parser import parse_query
        from src.search_part1 import search_part1
        parsed = parse_query(fq)
        with st.expander('Parsed query intent', expanded=False):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Dimensions', str(parsed.get('dimensions') or '-'))
            m2.metric('Shape', parsed.get('shape') or 'rectangle')
            m3.metric('Colors', ', '.join(parsed.get('color') or []) or '-')
            m4.metric('Styles', ', '.join(parsed.get('style') or []) or '-')
        with st.spinner('Searching...'):
            t0 = time.time()
            res = search_part1(fq, csv_path=ROOT / 'data' / 'products.csv', top_k=top_k)
            el = time.time() - t0
        if not res:
            st.markdown('<div class="bwarn">No results - try a broader query.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="bok">Found <strong>{len(res)}</strong> results in'
                f' <strong>{el:.2f}s</strong> - images proxied server-side</div>',
                unsafe_allow_html=True)
            render_grid(res, mode='part1', cols=3)
    elif go1 and not fq:
        st.markdown('<div class="bwarn">Please enter a query.</div>', unsafe_allow_html=True)

# TAB 2
with tab2:
    st.markdown('### Multimodal Image + Text Search')
    st.caption('CLIP (ViT-B/32) => FAISS => fusion: 0.6 x img + 0.4 x txt')
    ci2, cc2 = st.columns([1, 1])
    with ci2:
        st.markdown('#### Room Image')
        srcc = st.radio('Source', ['Upload photo', 'Sample room images'], horizontal=True)
        up_img = None; sp = None
        if srcc == 'Upload photo':
            uf = st.file_uploader('Upload', type=['jpg', 'jpeg', 'png', 'webp'],
                                  label_visibility='collapsed')
            if uf:
                from PIL import Image as PILImg
                up_img = PILImg.open(BytesIO(uf.read())).convert('RGB')
                st.image(up_img, use_container_width=True, caption='Your room')
        else:
            idir = ROOT / 'images'
            seen2: set = set(); uniq = []
            for f in sorted(idir.iterdir()):
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png'} and f.stem.lower() not in seen2:
                    seen2.add(f.stem.lower()); uniq.append(f)
            if uniq:
                cn = st.selectbox('Sample room', [f.name for f in uniq])
                sp = str(idir / cn)
                st.image(sp, use_container_width=True, caption=cn)
            else:
                st.warning('No sample images in images/')
    with cc2:
        st.markdown('#### Text Query (optional)')
        query2 = st.text_input('Refine', placeholder='e.g. warm traditional | 8x10',
                               label_visibility='collapsed')
        incl_ex = st.checkbox('Include match explanations', value=True)
        hi = bool(up_img or sp); ht = bool(query2.strip())
        ml = ('B - Image+Text (0.6 x img + 0.4 x txt)' if hi and ht
              else ('A - Image only' if hi else ('C - Text only' if ht else '-')))
        st.info(f'Fusion mode: {ml}')
    go2 = st.button('Find Matching Rugs', type='primary', use_container_width=True)
    if go2:
        ii = up_img or sp or None; ti = query2.strip()
        if ii is None and not ti:
            st.markdown('<div class="bwarn">Please provide an image or text query.</div>',
                        unsafe_allow_html=True)
        else:
            from src.search_part2 import search_multimodal
            with st.spinner('Running CLIP + FAISS... (first run builds index ~30s)'):
                t0 = time.time()
                try:
                    res2 = search_multimodal(ii, text_query=ti, top_k=top_k,
                                             csv_path=ROOT / 'data' / 'products.csv',
                                             include_explanations=incl_ex)
                    el2 = time.time() - t0
                except RuntimeError as e:
                    st.error(f'CLIP unavailable: {e}'); res2 = []; el2 = 0.
            if not res2:
                st.markdown('<div class="bwarn">No results found.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="bok">Found <strong>{len(res2)}</strong> results in'
                    f' <strong>{el2:.2f}s</strong></div>', unsafe_allow_html=True)
                top = res2[0]
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Top Fusion', f"{top['fusion_score']:.3f}")
                m2.metric('Img Sim', f"{top['image_similarity']:.3f}")
                m3.metric('Txt Sim', f"{top['text_similarity']:.3f}")
                m4.metric('Mode', top.get('mode', '-'))
                st.divider()
                render_grid(res2, mode='part2', cols=3)

# TAB 3
with tab3:
    st.markdown('### Catalog Explorer')
    st.caption(f'{len(catalog)} products loaded')
    import pandas as pd
    df = pd.DataFrame(catalog)[['handle', 'title', 'size', 'price', 'tags', 'color', 'style', 'image_url']]
    fc1, fc2, fc3 = st.columns(3)
    pmin = float(df['price'].dropna().min()); pmax = float(df['price'].dropna().max())
    prng = fc1.slider('Price ($)', pmin, pmax, (pmin, pmax))
    stxt = fc2.text_input('Filter title/tags', placeholder='e.g. geometric...')
    szf = fc3.text_input('Filter size', placeholder='e.g. 8x10, Runner...')
    filt = df.copy()
    filt = filt[filt['price'].fillna(0).between(*prng)]
    if stxt:
        filt = filt[filt['title'].str.contains(stxt, case=False, na=False) |
                    filt['tags'].str.contains(stxt, case=False, na=False)]
    if szf:
        filt = filt[filt['size'].str.contains(szf, case=False, na=False)]
    st.markdown(f'**{len(filt)}** products match')
    if len(filt) <= 30:
        st.markdown('#### Product Gallery')
        gc = st.columns(6)
        for ci3, (_, row) in enumerate(filt.iterrows()):
            url3 = row.get('image_url') or ''
            title3 = str(row.get('title') or ''); price3 = row.get('price')
            p3 = f'${price3:,.0f}' if price3 else '-'
            src3 = img_proxy(url3) if url3 else ''
            with gc[ci3 % 6]:
                itag = f'<img src="{src3}" style="width:100%;height:110px;object-fit:cover;"/>' if src3 else '&#x1F9F6;'
                st.markdown(
                    f'<div style="border-radius:10px;overflow:hidden;background:#1a1a2e;margin-bottom:4px;">{itag}</div>'
                    f'<div style="font-size:.68rem;color:#9ca3af;margin-bottom:8px;">{title3[:22]}<br/>'
                    f'<strong style="color:#34d399">{p3}</strong></div>',
                    unsafe_allow_html=True)
        st.divider()
    st.dataframe(filt.drop(columns=['image_url']), use_container_width=True, height=360)
    st.markdown('#### Price Distribution')
    try:
        import plotly.express as px
        fig = px.histogram(filt['price'].dropna(), nbins=25,
                           color_discrete_sequence=['#764ba2'], template='plotly_dark',
                           labels={'value': 'Price ($)', 'count': 'Products'})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                          plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
                          bargap=0.06, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(filt['price'].dropna())
    st.markdown('#### Top 20 Tags')
    from collections import Counter
    all_tags: list = []
    for t in filt['tags'].dropna():
        all_tags.extend(x.strip() for x in str(t).split(',') if x.strip())
    if all_tags:
        td = pd.DataFrame(Counter(all_tags).most_common(20), columns=['Tag', 'Count'])
        try:
            import plotly.express as px
            fig2 = px.bar(td, x='Count', y='Tag', orientation='h',
                          color='Count', color_continuous_scale='Purples', template='plotly_dark')
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               plot_bgcolor='#0f0f1a', paper_bgcolor='#0f0f1a',
                               yaxis=dict(autorange='reversed'), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.dataframe(td, use_container_width=True)
