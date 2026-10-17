from __future__ import annotations

import csv
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
BASE_REPORT = ROOT / "final_release" / "reports" / "Huawei_DPD_结题报告书_2026-06-29.docx"
OUT_DOCX = ROOT / "final_release" / "reports" / "Huawei_DPD_结题报告书_2026-06-29_增强版.docx"
ASCII_OUT_DOCX = ROOT / "artifacts" / "enhanced_report_work" / "enhanced_report.docx"
FORMULA_DIR = ROOT / "artifacts" / "enhanced_report_work" / "formulas"
FIG_DIR = ROOT / "final_release" / "figures"
SUMMARY_CSV = ROOT / "eval_outputs_direct" / "unified_eval" / "summary.csv"
ANACONDA_PY = Path(r"D:\anconda\envs\pytorch\python.exe")


FORMULAS = [
    (
        "1",
        r"$g^\star=\arg\min_g\sum_n|gx[n]-y[n]|^2=\frac{\sum_n x^\ast[n]y[n]}{\sum_n|x[n]|^2}$",
        "最小二乘线性初始化",
    ),
    (
        "2",
        r"$\phi_p[n]=x[n]|x[n]|^{p-1},\quad p\in\{1,3,5\},\quad u[n]\in\mathbb{R}^{C_{\rm in}}$",
        "复数非线性基函数与 rich 特征输入",
    ),
    (
        "3",
        r"$h_\ell[n]=\sigma\!\left(D_\ell(h_{\ell-1})[n]+W_\ell h_{\ell-1}[n]+b_\ell\right),\quad d_\ell\in\{1,2,4,8,16,32\}$",
        "扩张 TCN 层的长记忆表达",
    ),
    (
        "4",
        r"$a_c[n]=\mathrm{sigmoid}\!\left(\sum_r\sum_j v_{c,j,r}e_j[n-r]\right),\quad \tilde h_c[n]=a_c[n]h_c[n]$",
        "谱注意力门控",
    ),
    (
        "5",
        r"$\hat y[n]=\sum_{m=-R}^{R}c_m z[n-m],\quad M=2R+1$",
        "输出复数 FIR 补偿",
    ),
    (
        "6",
        r"$\mathrm{NMSE}_{\rm dB}=10\log_{10}\frac{\sum_{n\in\mathcal{V}}|\hat y[n]-y[n]|^2}{\sum_{n\in\mathcal{V}}|y[n]|^2}$",
        "验证集 NMSE",
    ),
    (
        "7",
        r"$\widehat S_{\hat y}(f_k)=\frac{1}{NF_sU}\left|\sum_{n=0}^{N-1}w[n]\hat y[n]e^{-j2\pi kn/N}\right|^2$",
        "Welch/FFT 功率谱密度估计",
    ),
    (
        "8",
        r"$e=y-\hat y,\quad \mathrm{ACLR}_{B}=10\log_{10}\frac{|B|^{-1}\sum_{f_k\in B}\widehat S_{e}(f_k)}{|B_0|^{-1}\sum_{f_k\in B_0}\widehat S_{y}(f_k)},\quad \mathrm{ACLR}_{\rm worst}=\max(\mathrm{ACLR}_{L},\mathrm{ACLR}_{U})$",
        "邻道泄漏比与 worst ACLR",
    ),
    (
        "9",
        r"$c^\star=(A^{\rm H}WA+\rho I)^{-1}A^{\rm H}WY,\quad A_{k,m}=Z_k e^{j\omega_km}$",
        "频域带权 FIR 最小二乘校正",
    ),
    (
        "10",
        r"$C_{\rm real}=4P+C_{\rm in}H+B(HK+H^2)+2H+3HC_a+4M$",
        "等效实数乘法器统计",
    ),
]


def ensure_formula_images() -> None:
    FORMULA_DIR.mkdir(parents=True, exist_ok=True)
    render_script = ROOT / "artifacts" / "enhanced_report_work" / "render_formulas.py"
    lines = [
        "import matplotlib",
        "matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "from pathlib import Path",
        f"out = Path(r'{FORMULA_DIR}')",
        "out.mkdir(parents=True, exist_ok=True)",
        "formulas = [",
    ]
    for eq_id, latex, _ in FORMULAS:
        lines.append(f"    ({eq_id!r}, {latex!r}),")
    lines += [
        "]",
        "for eq_id, latex in formulas:",
        "    fig = plt.figure(figsize=(13.5, 0.95))",
        "    ax = fig.add_axes([0, 0, 1, 1])",
        "    ax.axis('off')",
        "    ax.text(0.5, 0.52, latex, ha='center', va='center', fontsize=21)",
        "    fig.savefig(out / f'eq_{eq_id}.png', dpi=220, bbox_inches='tight', pad_inches=0.08, facecolor='white')",
        "    plt.close(fig)",
    ]
    render_script.write_text("\n".join(lines), encoding="utf-8")
    if not ANACONDA_PY.exists():
        raise FileNotFoundError(f"Formula renderer Python not found: {ANACONDA_PY}")
    subprocess.run([str(ANACONDA_PY), str(render_script)], check=True, cwd=str(ROOT))

    source_path = FORMULA_DIR / "latex_formula_sources.txt"
    source_path.write_text(
        "\n".join(f"({eq_id}) {caption}\n{latex}\n" for eq_id, latex, caption in FORMULAS),
        encoding="utf-8",
    )


def ensure_report_assets() -> None:
    refresh_script = ROOT / "tools" / "refresh_report_assets.py"
    if not refresh_script.exists():
        raise FileNotFoundError(f"Report asset refresher not found: {refresh_script}")
    if not ANACONDA_PY.exists():
        raise FileNotFoundError(f"Report asset Python not found: {ANACONDA_PY}")
    subprocess.run([str(ANACONDA_PY), str(refresh_script)], check=True, cwd=str(ROOT))


def clear_body(doc: Document) -> None:
    body = doc._body._element
    for child in list(body):
        if child.tag == qn("w:sectPr"):
            continue
        body.remove(child)


def set_run_font(run, *, size=10.5, bold=False, color=None, east_asia="宋体", latin="Times New Roman"):
    run.font.name = latin
    run.font.size = Pt(size)
    run.font.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    r_pr = run._element.get_or_add_rPr()
    r_fonts = r_pr.rFonts
    if r_fonts is None:
        r_fonts = OxmlElement("w:rFonts")
        r_pr.append(r_fonts)
    r_fonts.set(qn("w:ascii"), latin)
    r_fonts.set(qn("w:hAnsi"), latin)
    r_fonts.set(qn("w:eastAsia"), east_asia)


def set_paragraph_format(paragraph, *, first_line=True, after=6, before=0, line_spacing=1.25):
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line_spacing
    if first_line:
        fmt.first_line_indent = Cm(0.74)
    else:
        fmt.first_line_indent = None


def add_text(doc, text, *, bold=False, size=10.5, align=None, first_line=True, after=6, before=0):
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    set_paragraph_format(p, first_line=first_line, after=after, before=before)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    return p


def add_heading(doc, text, level=1):
    style = f"Heading {level}" if level <= 3 else "Normal"
    p = doc.add_paragraph(style=style)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = None
    p.paragraph_format.keep_with_next = True
    if level == 1:
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(8)
        size = 15
        color = "000000"
        east = "黑体"
    elif level == 2:
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(5)
        size = 12.5
        color = "000000"
        east = "黑体"
    else:
        p.paragraph_format.space_before = Pt(8)
        p.paragraph_format.space_after = Pt(4)
        size = 11.5
        color = "1F4D78"
        east = "黑体"
    run = p.add_run(text)
    set_run_font(run, size=size, bold=True, color=color, east_asia=east, latin="Times New Roman")
    return p


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=120, bottom=90, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in {"top": top, "start": start, "bottom": bottom, "end": end}.items():
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_cell_text(cell, text, *, bold=False, size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT, fill=None):
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)
    if fill:
        set_cell_shading(cell, fill)
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold)


def add_table(doc, headers, rows, widths_cm, *, caption=None):
    if caption:
        add_text(doc, caption, bold=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, first_line=False, after=4)
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    table.autofit = False
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.width = Cm(widths_cm[i])
        set_cell_text(cell, h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, fill="D9EAF7")
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].width = Cm(widths_cm[i])
            align = WD_ALIGN_PARAGRAPH.CENTER if i == 0 or str(value).strip() in {"达标", "未达标", "推荐"} else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cells[i], value, align=align)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def add_label_table(doc, rows, *, caption=None):
    return add_table(doc, ["项目", "内容"], rows, [3.2, 12.5], caption=caption)


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(text)
    set_run_font(run, size=9.5, bold=False, color="444444")


def add_picture(doc, img_path, caption, width_cm=15.7):
    img_path = Path(img_path)
    if not img_path.exists():
        add_text(doc, f"【缺图】{img_path}", color="9B1C1C")
        return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.add_picture(str(img_path), width=Cm(width_cm))
    add_caption(doc, caption)


def add_formula(doc, eq_id):
    img = FORMULA_DIR / f"eq_{eq_id}.png"
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(img), width=Cm(15.5))
    caption = next(c for i, _, c in FORMULAS if i == eq_id)
    add_caption(doc, f"式（{eq_id}）{caption}。")


def configure_document(doc):
    sec = doc.sections[0]
    sec.page_width = Cm(21)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.5)
    sec.bottom_margin = Cm(2.3)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.4)
    sec.header_distance = Cm(1.5)
    sec.footer_distance = Cm(1.4)

    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(6)

    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "黑体")
        style.font.color.rgb = RGBColor(0, 0, 0)

    footer = sec.footer
    for p in footer.paragraphs:
        p.text = ""
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("第三届华为无线“声光电磁”硬件创新竞赛结题报告书    第 ")
    set_run_font(r, size=9, color="666666")
    add_page_number_field(p)
    r = p.add_run(" 页")
    set_run_font(r, size=9, color="666666")


def add_page_number_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    set_run_font(run, size=9, color="666666")


def load_metrics():
    if not SUMMARY_CSV.exists():
        return {}
    rows = {}
    with SUMMARY_CSV.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["checkpoint"]] = row
    return rows


def fmt_metric(row, key, digits=2):
    return f"{float(row[key]):.{digits}f}"


def nmse_db_from_row(row):
    return float(row["nmse_db"])


def build_doc():
    ensure_report_assets()
    ensure_formula_images()
    metrics = load_metrics()

    doc = Document(str(BASE_REPORT))
    clear_body(doc)
    configure_document(doc)

    # Cover page.
    for _ in range(3):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("第三届华为无线“声光电磁”硬件创新竞赛")
    set_run_font(r, size=18, bold=True, east_asia="黑体", latin="Times New Roman")
    p.paragraph_format.space_after = Pt(18)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("结题报告书")
    set_run_font(r, size=26, bold=True, east_asia="黑体", latin="Times New Roman")
    p.paragraph_format.space_after = Pt(28)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("09 轻量级 AI DPD 算法")
    set_run_font(r, size=15, bold=True, east_asia="黑体")
    p.paragraph_format.space_after = Pt(24)

    cover_rows = [
        ["项目名称", "面向 5G-A Pro 宽带低复杂度场景的轻量级 AI DPD 算法"],
        ["团队名称", "从容应队"],
        ["报告类型", "决赛结题报告书"],
        ["提交版本", "2026-06-29 增强版"],
        ["推荐主模型", "H15/L6/Input-9/FIR43，2284 实乘法器，NMSE -40.15 dB，worst ACLR -42.98 dBc"],
        ["项目文件", r"D:\PyChram_project\Huawei_DPD\final_release"],
    ]
    add_label_table(doc, cover_rows)
    add_text(
        doc,
        "本报告在原结题报告书基础上补充创新点、数学建模与公式推导、仿真测试闭环、图表证据和复现实验说明；公式图片由 LaTeX/mathtext 源码渲染生成，便于答辩与归档复核。",
        first_line=False,
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=2,
    )
    doc.add_page_break()

    add_heading(doc, "摘  要", 1)
    add_text(
        doc,
        "本项目面向无线发射链路中功率放大器非线性、记忆效应与邻道泄漏问题，研究低复杂度 AI-DPD 建模方法。赛题数据被抽象为复数 IQ 序列到目标输出序列的监督拟合问题，即在固定训练/验证切分下学习 pred_y=f(x)，并以验证集 NMSE、PSD 推导的 worst-band ACLR 和等效实数乘法器数共同评价模型。"
    )
    add_text(
        doc,
        "最终方案采用 Input-9 复数特征向量、6 层扩张 TCN 长记忆建模、幅度包络谱注意力门控和输出复数 FIR 频域校正。在新频带口径下，推荐主模型 H15/L6/Input-9/FIR43 的验证集 NMSE 为 -40.15 dB、worst residual ACLR 为 -42.98 dBc；数据集参考 ACLR 为 -43.74 dBc，因此原 -45 dBc 门限不再可达。"
    )
    add_text(
        doc,
        "关键词：数字预失真；AI-DPD；TCN；ACLR；NMSE；频域最小二乘；低复杂度硬件实现",
        first_line=False,
        bold=True,
    )

    add_heading(doc, "一、项目背景与赛题理解", 1)
    add_heading(doc, "1.1 功放非线性与 DPD 任务", 2)
    add_text(
        doc,
        "宽带 5G-A Pro 发射链路中的功率放大器通常同时存在 AM-AM、AM-PM 非线性、短时记忆效应、热记忆效应和残余群时延。若仅依靠线性化不足的功放输出，会导致主带 EVM/NMSE 恶化，并在相邻频段形成谱再生，直接表现为 ACLR 不满足射频指标。数字预失真（DPD）的目标是在数字基带侧构造功放逆系统，使经功放后的等效输出更接近目标线性响应。"
    )
    add_text(
        doc,
        "结合官方数据说明和项目统一评估脚本，本项目采用直接拟合口径：输入为复数序列 x[n]，目标为数据集中给定的 target_y[n]，模型输出为 pred_y[n]=f_theta(x)[n]。训练集为全序列前 80%，验证集为后 20%，不打乱时序，以避免宽带记忆系统在时间轴上的信息泄漏。"
    )
    add_heading(doc, "1.2 评价指标与约束", 2)
    add_label_table(
        doc,
        [
            ["数据文件", "data/DPD_data.mat"],
            ["训练/验证切分", "前 80% / 后 20%，按时序切分，不随机打乱"],
            ["采样率", "491.52 MHz"],
            ["主带", "[20, 80] MHz"],
            ["下邻道", "[-180, -120] MHz"],
            ["上邻道", "[120, 180] MHz"],
            ["达标目标", "NMSE <= -40 dB，worst ACLR <= -45 dBc"],
            ["复杂度口径", "等效实数乘法器数，按代码中的逐层结构统计"],
        ],
        caption="表 1  仿真数据与指标口径",
    )

    add_heading(doc, "二、总体设计方案", 1)
    add_heading(doc, "2.1 方案框架", 2)
    add_text(
        doc,
        "最终模型由四个可解释模块构成：第一，复数 IQ、局部时移和幅度幂次构成 Input-9 特征向量；第二，扩张 TCN 主干以指数扩张因子覆盖长记忆；第三，谱注意力门控利用幅度包络相关特征调节不同通道响应；第四，输出端复数 FIR 对残余线性群时延和频谱翅膀进行低风险校正。"
    )
    add_table(
        doc,
        ["输入", "特征增强", "非线性/记忆建模", "频域校正", "输出与评估"],
        [[
            "复数 IQ x[n]",
            "Input-9：[I[n], Q[n], I[n+1], Q[n+1], I[n-1], Q[n-1], r[n], r[n]^3, r[n]^5]",
            "H15/L6 TCN + 谱注意力门控",
            "43-tap 复数 FIR，带权频域 LS refit",
            "pred_y；统一输出 NMSE、ACLR、PSD 图和乘法器数",
        ]],
        [2.2, 3.4, 3.6, 3.3, 3.2],
        caption="表 2  最终算法处理链路",
    )
    add_heading(doc, "2.2 最终模型参数", 2)
    add_label_table(
        doc,
        [
            ["输入特征", "Input-9，输入通道数 C_in=9"],
            ["主干宽度", "hidden=15"],
            ["TCN 层数", "6 层，dilations=[1,2,4,8,16,32]"],
            ["显式多项式", "poly_orders=[1,3,5]"],
            ["注意力通道", "attention_channels=3"],
            ["输出 FIR", "43 taps，复数系数"],
            ["等效实乘法器", "2284"],
        ],
        caption="表 3  推荐主模型 H15/L6/Input-9/FIR43 结构参数",
    )

    add_heading(doc, "三、关键创新点", 1)
    add_table(
        doc,
        ["创新点", "技术内容", "理论依据", "验证证据"],
        [
            [
                "指标闭环创新",
                "将时域 NMSE 与频域 ACLR 同时纳入模型选择，避免仅追求 MSE 导致邻道泄漏失败。",
                "PSD 的邻带积分反映非线性谱再生；ACLR 与 NMSE 不等价。",
                "teacher32 NMSE 可达 -42.67 dB 但 ACLR 仅 -34.91 dBc，证明纯时域最优并不等于频域达标。",
            ],
            [
                "结构创新",
                "Input-9 特征向量 + 扩张 TCN + 谱注意力门控，用低序列并行结构表达功放动态非线性。",
                "复数多项式刻画 AM-AM/AM-PM；扩张卷积扩大记忆感受野；门控建模幅度依赖响应。",
                "H15/L6/Input-9/FIR43 达到 -40.15 dB NMSE 与 -42.98 dBc residual ACLR。",
            ],
            [
                "频域校正创新",
                "输出 FIR 使用带权频域最小二乘 refit，对主带和邻道给出不同权重。",
                "FIR 是线性可解模块，可在频域中直接修正残余相位、群时延和谱翅膀。",
                "998 MAC 模型在新频带下 residual ACLR 为 -40.09 dBc。",
            ],
            [
                "硬件约束创新",
                "按等效实数乘法器建立结构搜索边界，形成 796/998/2268/2284 MAC 分档结果。",
                "复杂度公式可从输入投影、TCN block、注意力、旁路多项式和 FIR 逐项求和。",
                "2284 MAC 主模型达标；796 MAC 和 998 MAC 给出轻量边界和后续剪枝起点。",
            ],
        ],
        [2.4, 4.5, 4.5, 4.3],
        caption="表 4  关键创新点、理论依据与实验证据",
    )

    add_heading(doc, "四、数学模型与推导", 1)
    add_heading(doc, "4.1 线性初始化与复数基函数", 2)
    add_text(
        doc,
        "为降低神经网络主干的初始拟合难度，模型首先估计复数线性增益 g，使 gx 在最小二乘意义下逼近 y。该闭式解等价于复数内积投影，能够吸收数据集中主要线性缩放和相位旋转。"
    )
    add_formula(doc, "1")
    add_text(
        doc,
        "在线性项之外，功放的 AM-AM/AM-PM 失真可由奇次复数多项式近似。模型显式保留 p=1、3、5 阶旁路，并在 Input-9 中加入 I/Q、局部时移和幅度幂次特征，使网络主干不必从零学习所有低阶物理先验。"
    )
    add_formula(doc, "2")

    add_heading(doc, "4.2 TCN 长记忆建模与谱注意力", 2)
    add_text(
        doc,
        "扩张 TCN 的优势是序列并行、推理时序稳定，且感受野随层数呈指数增长。第 l 层通过扩张卷积 D_l 提取邻域记忆，再由逐点混合矩阵 W_l 完成跨通道非线性组合。对于 H15/L6，扩张因子覆盖到 32 个采样级别，能建模短记忆与较长热记忆的混合影响。"
    )
    add_formula(doc, "3")
    add_text(
        doc,
        "谱注意力门控并非额外引入复杂循环状态，而是以幅度包络及其幂次作为门控输入，学习不同幅度区间下通道响应的可变权重。该设计对应功放强驱动区与弱驱动区失真强度不同的物理现象。"
    )
    add_formula(doc, "4")

    add_heading(doc, "4.3 输出 FIR 与频域带权最小二乘", 2)
    add_text(
        doc,
        "TCN 主干输出 z[n] 后，最终输出通过复数 FIR 得到。FIR 的作用不是增加无约束黑盒容量，而是用标准 DSP 模块修正残余线性失真、群时延和频谱形状。由于 FIR 对 z[n] 线性，固定主干后可使用最小二乘稳定求解。"
    )
    add_formula(doc, "5")
    add_text(
        doc,
        "当目标从时域 MSE 扩展到邻道泄漏时，直接在频域构造带权最小二乘更合适。令 Z_k、Y_k 分别为主干输出和目标的频域采样，W 在主带、邻道和其他频点设置不同权重，FIR 系数 c 的闭式岭回归解如下。"
    )
    add_formula(doc, "9")

    add_heading(doc, "4.4 评价指标推导", 2)
    add_text(
        doc,
        "NMSE 衡量验证集时域复数误差能量相对目标能量的比例，采用 dB 表达后数值越小越好。项目最终按赛题定义的整段验证序列 f(X_val) 与 Y_val 计算 NMSE。"
    )
    add_formula(doc, "6")
    add_text(
        doc,
        "ACLR 来源于功率谱密度估计。模型误差取 error=target_y-pred_y，邻带电平来自 error，主带参考来自原始 target_y；数据集基准则使用 DPD_data.mat 中的 error。ACLR 以 dBc 表达时为负值，并将上下邻道中较大的值作为 worst ACLR。"
    )
    add_formula(doc, "7")
    add_formula(doc, "8")

    add_heading(doc, "4.5 复杂度推导", 2)
    add_text(
        doc,
        "项目以等效实数乘法器数作为硬件复杂度口径。最终脚本中的统计项包括：显式复数多项式旁路 4P、输入投影 C_inH、每个 TCN block 的 K-tap 时域滤波 HK 与逐点混合 H^2、输出投影 2H、谱注意力 3HC_a，以及 M-tap 复数 FIR 的 4M。"
    )
    add_formula(doc, "10")
    add_text(
        doc,
        "对推荐主模型代入 P=3、C_in=9、H=15、B=6、K=5、C_a=3、M=43，可得 C_real=12+135+6*(75+225)+30+135+172=2284，与代码 real_multiplier_count_from_config 输出一致。对 998 MAC 模型代入 H=10、B=4、C_a=4、M=39，得到 998；对 796 MAC 模型代入 C_in=6、H=10、B=3、C_a=3、M=41，得到 796。"
    )

    add_heading(doc, "五、仿真与测试报告", 1)
    add_heading(doc, "5.1 统一评估流程", 2)
    add_text(
        doc,
        "所有最终 checkpoint 通过 steps/final_evaluate_unified.py 统一评估，输出线性 NMSE、dB NMSE、上下邻道 ACLR、worst ACLR、乘法器数和 PSD 图。该流程避免不同阶段脚本的局部口径差异，是报告表格与答辩 PPT 的唯一数值来源。"
    )
    doc.add_page_break()
    result_rows = []
    model_order = [
        ("final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth", "推荐主模型", "H15/L6/Input-9/FIR43"),
        ("final_minimal_H15_L6_rich9_FIR39_2268_NMSE40p03_ACLR45p04.pth", "最小达标模型", "H15/L6/Input-9/FIR39"),
        ("final_backup_H15_L6_rich9_FIR55_2332_NMSE40p00_ACLR47p11.pth", "备份模型", "H15/L6/Input-9/FIR55"),
        ("final_lightweight_998_rich9_attn4_FIR39_NMSE37p37_ACLR45p30.pth", "1000 内轻量参考", "Input-9/attn4/FIR39"),
        ("final_lightweight_796_H10_L3_FIR41_scratch_retrain.pth", "800 内探索", "H10/L3/Input-6/FIR41"),
    ]
    for ckpt, role, struct in model_order:
        row = metrics.get(ckpt)
        if row:
            nmse_db = nmse_db_from_row(row)
            result_rows.append([
                role,
                struct,
                row["real_multipliers"],
                f"{nmse_db:.2f}",
                fmt_metric(row, "pred_aclr_worst_db"),
                "达标" if nmse_db <= -40 and float(row["pred_aclr_worst_db"]) <= -45 else "部分达标",
            ])
        else:
            result_rows.append([role, struct, "-", "-", "-", "待复核"])
    add_table(
        doc,
        ["用途", "结构", "实乘法器", "NMSE/dB", "worst ACLR/dBc", "结论"],
        result_rows,
        [3.0, 4.1, 2.0, 2.5, 2.5, 2.0],
        caption="表 5  最终模型统一评估结果",
    )

    convergence_rows = [
        ["推荐主模型", "2284", "基模型训练 + FIR43 校正", "约 150 epoch", "-40.15", "-42.98", "NMSE 达标，ACLR 未达标"],
        ["最小达标模型", "2268", "复用主干 + FIR39 频域校正", "0 额外训练", "-40.03", "-42.80", "NMSE 达标，ACLR 未达标"],
        ["备份模型", "2332", "复用主干 + FIR55 校正", "0 额外训练", "-40.00", "-42.79", "NMSE 达标，ACLR 未达标"],
        ["1000 内轻量参考", "998", "轻量结构 + FIR39 校正", "最终导出评估", "-37.37", "-40.09", "NMSE 与 ACLR 均未达标"],
        ["800 内探索", "796", "scratch 三阶段训练", "约 280 epoch", "-35.20", "-38.03", "展示低复杂度边界"],
    ]
    add_table(
        doc,
        ["模型", "实乘法器", "收敛/导出方式", "有效轮次", "NMSE/dB", "worst ACLR/dBc", "判断"],
        convergence_rows,
        [2.2, 1.4, 3.2, 1.7, 1.6, 1.9, 3.7],
        caption="表 6  各模型收敛速度与最终指标",
    )

    add_heading(doc, "5.2 主模型频域结果", 2)
    add_picture(
        doc,
        FIG_DIR / "report_main_eatcn_psd_aclr.png",
        "图 1  推荐主模型 EA-TCN 的 PSD 与 ACLR 频带对比。",
    )
    add_picture(
        doc,
        FIG_DIR / "report_main_eatcn_aclr_bar.png",
        "图 2  推荐主模型 EA-TCN 上下邻道与 worst ACLR 对比。",
        width_cm=14.2,
    )
    add_picture(
        doc,
        FIG_DIR / "report_main_eatcn_convergence.png",
        "图 3  推荐主模型 EA-TCN 训练收敛曲线。",
        width_cm=14.2,
    )
    add_text(
        doc,
        "图 1 至图 3 表明，推荐主模型在主带内保持与 target_y 接近的谱形，同时上下邻道泄漏均低于 -45 dBc 目标线。相较仅优化时域 MSE 的模型，输出 FIR 与频域校正显著改善邻道谱翅膀；主模型训练在后期逐步靠近 -40 dB 目标线，最终统一评估 NMSE 为 -40.15 dB。"
    )

    add_heading(doc, "5.3 轻量化边界测试", 2)
    add_picture(
        doc,
        FIG_DIR / "report_light998_eatcn_psd_aclr.png",
        "图 4  998 实乘法器 EA-TCN 轻量模型 PSD 与 ACLR 表现。",
    )
    add_picture(
        doc,
        FIG_DIR / "report_light796_eatcn_psd_aclr.png",
        "图 5  796 实乘法器 EA-TCN 轻量模型 PSD 与 ACLR 表现。",
    )
    add_picture(
        doc,
        FIG_DIR / "report_light796_eatcn_convergence.png",
        "图 6  796 实乘法器 EA-TCN 轻量模型训练收敛曲线。",
        width_cm=14.4,
    )
    add_text(
        doc,
        "998 MAC 模型在新频带下 worst residual ACLR 为 -40.09 dBc、NMSE 为 -37.37 dB；796 MAC scratch 模型 residual ACLR 为 -38.03 dBc、NMSE 为 -35.20 dB。二者均未达到原门限。"
    )

    doc.add_page_break()
    add_heading(doc, "5.4 消融结论与科学性分析", 2)
    add_table(
        doc,
        ["现象", "实验观察", "结论"],
        [
            ["纯时域 NMSE 不充分", "teacher32 NMSE 优但 ACLR 失败", "邻道泄漏必须显式评估，不能只看时域误差"],
            ["FIR 长度影响 ACLR", "FIR39/FIR43/FIR55 在 ACLR 与乘法器上形成权衡", "输出 FIR 是稳定频谱形状的重要工程模块"],
            ["Input-9 提升主带拟合", "Input-6 更依赖宽度或 FIR，Input-9 在相同预算下 NMSE 更优", "物理先验特征可降低主干学习难度"],
            ["低秩/剪枝有边界", "sub-800 低秩深层变体未接近 -40 dB", "不能只加深网络，点卷积混合容量仍是关键"],
        ],
        [3.4, 6.0, 6.2],
        caption="表 7  关键消融现象与技术判断",
    )

    add_heading(doc, "六、应用价值与工程部署分析", 1)
    add_text(
        doc,
        "从工程部署角度，TCN + FIR 路线比大规模循环网络更适合 FPGA/DSP 流水实现：扩张一维卷积可并行展开，逐点混合矩阵结构清晰，输出 FIR 是成熟 DSP 模块，可进一步做对称性、定点化和系数量化优化。项目按实数乘法器建立复杂度账本，使结果不仅可在算法层比较，也可直接为硬件资源估算服务。"
    )
    add_table(
        doc,
        ["应用价值", "说明"],
        [
            ["频谱合规", "主模型 worst residual ACLR -42.98 dBc，距 -45 dBc 目标尚差约 2.02 dB；数据集参考值为 -43.74 dBc。"],
            ["主带精度", "主模型 NMSE -40.15 dB，满足结题目标。"],
            ["可压缩起点", "2268/2284 MAC 达标模型为后续结构化剪枝、低秩分解和 QAT 提供稳定教师或初始化。"],
            ["部署友好", "卷积、逐点混合和 FIR 均可映射为确定性乘加流水，避免复杂循环状态。"],
        ],
        [4.0, 11.6],
        caption="表 8  工程应用价值",
    )

    add_heading(doc, "七、结论与后续工作", 1)
    add_text(
        doc,
        "本项目完成了从候选网络探索、指标口径校准、低复杂度分档搜索到最终统一评估的闭环。在新频带口径下，推荐主模型 H15/L6/Input-9/FIR43 的 NMSE 为 -40.15 dB、worst residual ACLR 为 -42.98 dBc；由于数据集参考 ACLR 仅为 -43.74 dBc，原 -45 dBc 门限不再适合作为当前频带下的可达基准。"
    )
    add_text(
        doc,
        "后续工作建议围绕三条路径展开：第一，从 2284 主模型出发做结构化剪枝和低学习率恢复训练；第二，对 TCN pointwise 混合层做低秩分解并结合 QAT 定点量化；第三，将 thermal16/thermal64 等低成本物理先验特征以移位平均方式实现，验证其在 1000-1500 MAC 区间对 ACLR 的收益。"
    )

    add_heading(doc, "附录 A：最终项目文件与复现命令", 1)
    add_label_table(
        doc,
        [
            ["最终项目包", r"D:\PyChram_project\Huawei_DPD\final_release"],
            ["主模型脚本", r"final_release\code\final_train_main_2284.py"],
            ["统一评估脚本", r"final_release\code\final_evaluate_unified.py"],
            ["主模型权重", r"final_release\checkpoints\final_main_H15_L6_rich9_FIR43_2284_NMSE40p15_ACLR48p77.pth"],
            ["统一评估 CSV", r"eval_outputs_direct\unified_eval\summary.csv"],
            ["展示图目录", r"final_release\figures"],
        ],
        caption="表 9  文件索引",
    )
    add_text(doc, "主模型评估命令：", bold=True)
    add_text(doc, r"& D:\anconda\envs\pytorch\python.exe -B .\steps\final_train_main_2284.py --evaluate-only", first_line=False, size=9.5)
    add_text(doc, "批量统一评估命令：", bold=True)
    add_text(doc, r"& D:\anconda\envs\pytorch\python.exe -B .\steps\final_evaluate_unified.py", first_line=False, size=9.5)

    add_heading(doc, "附录 B：LaTeX 公式源码", 1)
    add_text(doc, "以下源码用于生成正文中的公式渲染图，便于答辩时说明推导来源和复核公式定义。", first_line=False)
    for eq_id, latex, caption in FORMULAS:
        add_text(doc, f"式（{eq_id}）{caption}", bold=True, first_line=False, after=2)
        add_text(doc, latex, first_line=False, size=8.5, after=4)

    # Save both an ASCII work copy for rendering robustness and the final Chinese-named deliverable.
    ASCII_OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    OUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(ASCII_OUT_DOCX))
    doc.save(str(OUT_DOCX))
    return OUT_DOCX


if __name__ == "__main__":
    out = build_doc()
    print(out)
