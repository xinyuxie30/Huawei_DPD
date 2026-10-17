import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const projectRoot = process.env.PROJECT_ROOT || "D:/PyChram_project/Huawei_DPD";
const outDir = process.env.OUT_DIR || path.join(projectRoot, "final_release", "defense_pptx");
const qaDir = process.env.QA_DIR || path.join(outDir, "qa");
const finalPptx = process.env.FINAL_PPTX || path.join(outDir, "AI_DPD_final_defense_direct.pptx");

const W = 1280;
const H = 720;
const C = {
  navy: "#00407A",
  navy2: "#123F66",
  lightBlue: "#EAF3FB",
  green: "#12823B",
  lightGreen: "#EAF7EF",
  orange: "#C56A00",
  lightOrange: "#FFF1E3",
  red: "#B3261E",
  grayText: "#334155",
  gray: "#64748B",
  lightGray: "#F4F6F8",
  white: "#FFFFFF",
  line: "#D6DEE8",
};

const asset = (...parts) => path.join(projectRoot, ...parts);
const figures = asset("final_release", "figures");
const logo = asset("final_release", "defense_beamer", "pic", "LogoCQUPT.jpg");

const img = {
  mainPsd: path.join(figures, "final_main_H15_L6_rich9_FIR43_2284_psd_aclr.png"),
  mainAclr: path.join(figures, "final_main_H15_L6_rich9_FIR43_2284_aclr_bar.png"),
  mainConv: path.join(figures, "final_main_H15_L6_rich9_FIR43_2284_convergence.png"),
  light998: path.join(figures, "final_lightweight_998_rich9_attn4_FIR39_psd_aclr.png"),
  light796: path.join(figures, "final_lightweight_796_H10_L3_FIR41_scratch_retrain_psd_aclr.png"),
  light796Conv: path.join(figures, "final_lightweight_796_H10_L3_FIR41_scratch_retrain_convergence.png"),
  model786Psd: path.join(figures, "step800_scratch_h10_l4_base6_rank3_fir41_786_psd_aclr.png"),
  model786Aclr: path.join(figures, "step800_scratch_h10_l4_base6_rank3_fir41_786_aclr_bar.png"),
  model786Conv: path.join(figures, "step800_scratch_h10_l4_base6_rank3_fir41_786_convergence.png"),
  eatcn: asset("final_release", "defense_beamer", "figures", "eatcn_template.png"),
  tcn: asset("final_release", "defense_beamer", "figures", "tcn_dilation.png"),
  dpd: asset("final_release", "defense_beamer", "figures", "dpd_task.png"),
};

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function readImageBlob(imagePath) {
  const bytes = await fs.readFile(imagePath);
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}

function contentType(imagePath) {
  const ext = imagePath.toLowerCase();
  if (ext.endsWith(".jpg") || ext.endsWith(".jpeg")) return "image/jpeg";
  return "image/png";
}

function shape(slide, position, fill = "none", lineFill = "none", geometry = "rect") {
  return slide.shapes.add({
    geometry,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  });
}

function textbox(slide, text, position, style = {}) {
  const t = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  t.text = text;
  t.text.style = {
    fontSize: style.fontSize ?? 22,
    bold: style.bold ?? false,
    color: style.color ?? C.grayText,
    alignment: style.alignment ?? "left",
  };
  return t;
}

function header(slide, title, section = "") {
  shape(slide, { left: 0, top: 0, width: W, height: 78 }, C.navy, C.navy);
  textbox(slide, title, { left: 58, top: 20, width: 920, height: 48 }, {
    fontSize: 34,
    bold: true,
    color: C.white,
  });
  if (section) {
    textbox(slide, section, { left: 1000, top: 27, width: 220, height: 28 }, {
      fontSize: 15,
      color: "#D6E7F7",
      alignment: "right",
    });
  }
}

function footer(slide, n) {
  shape(slide, { left: 0, top: 688, width: W, height: 32 }, C.navy, C.navy);
  textbox(slide, "Lightweight AI DPD Final Defense", { left: 58, top: 694, width: 500, height: 20 }, {
    fontSize: 12,
    color: "#D6E7F7",
  });
  textbox(slide, `${n} / 16`, { left: 1160, top: 694, width: 70, height: 20 }, {
    fontSize: 12,
    color: "#D6E7F7",
    alignment: "right",
  });
}

function titleSlide(slide) {
  slide.background.fill = C.white;
  shape(slide, { left: 0, top: 0, width: W, height: 150 }, C.navy, C.navy);
  textbox(slide, "轻量级 AI DPD 算法结题答辩", { left: 78, top: 48, width: 860, height: 72 }, {
    fontSize: 52,
    bold: true,
    color: C.white,
  });
  textbox(slide, "第三届华为无线“声光电磁”硬件创新竞赛 09 赛题", { left: 84, top: 170, width: 700, height: 36 }, {
    fontSize: 25,
    color: C.navy,
  });
  textbox(slide, "基于 EA-TCN 的可解释、低复杂度、频域约束 DPD 方案", { left: 84, top: 232, width: 920, height: 44 }, {
    fontSize: 30,
    bold: true,
    color: C.grayText,
  });
  textbox(slide, "重庆邮电大学 通信与信息工程学院 · 2026年7月", { left: 84, top: 310, width: 660, height: 32 }, {
    fontSize: 20,
    color: C.gray,
  });
  shape(slide, { left: 84, top: 410, width: 680, height: 118 }, C.lightBlue, C.line);
  textbox(slide, "核心结论", { left: 112, top: 430, width: 160, height: 30 }, {
    fontSize: 24,
    bold: true,
    color: C.navy,
  });
  textbox(slide, "主模型 H15/L6/Input-9/FIR43 在 2284 个等效实乘法器下达到 NMSE -40.15 dB 与 worst ACLR -48.77 dBc。", { left: 112, top: 468, width: 620, height: 46 }, {
    fontSize: 21,
    color: C.grayText,
  });
  slide.images.add({ blob: awaitImage(logo), contentType: "image/jpeg", alt: "CQUPT logo", fit: "contain", position: { left: 1055, top: 505, width: 110, height: 110 } });
  footer(slide, 1);
}

let imageCache = new Map();
function awaitImage(imagePath) {
  throw new Error("awaitImage must be replaced in async creation.");
}

async function addImage(slide, imagePath, position, fit = "contain", alt = "") {
  let blob = imageCache.get(imagePath);
  if (!blob) {
    blob = await readImageBlob(imagePath);
    imageCache.set(imagePath, blob);
  }
  slide.images.add({
    blob,
    contentType: contentType(imagePath),
    alt,
    fit,
    position,
  });
}

function bullet(slide, items, x, y, w, fontSize = 21, color = C.grayText, gap = 40) {
  items.forEach((item, i) => {
    textbox(slide, "▶", { left: x, top: y + i * gap + 2, width: 24, height: 24 }, { fontSize: fontSize - 2, color: C.navy });
    textbox(slide, item, { left: x + 32, top: y + i * gap, width: w - 32, height: gap - 4 }, { fontSize, color });
  });
}

function metric(slide, label, value, note, x, y, w, color = C.green, bg = C.lightGreen) {
  shape(slide, { left: x, top: y, width: w, height: 106 }, bg, "none");
  textbox(slide, value, { left: x + 18, top: y + 14, width: w - 36, height: 34 }, {
    fontSize: 26,
    bold: true,
    color,
  });
  textbox(slide, `${label}${note ? ` · ${note}` : ""}`, { left: x + 18, top: y + 56, width: w - 36, height: 40 }, {
    fontSize: 14,
    color,
  });
}

function formula(slide, latex, x, y, w, h, title = "") {
  shape(slide, { left: x, top: y, width: w, height: h }, "#F8FAFC", C.line);
  if (title) {
    textbox(slide, title, { left: x + 18, top: y + 12, width: w - 36, height: 24 }, {
      fontSize: 16,
      bold: true,
      color: C.navy,
    });
  }
  textbox(slide, latex, { left: x + 18, top: y + (title ? 44 : 18), width: w - 36, height: h - (title ? 54 : 28) }, {
    fontSize: 18,
    color: "#111827",
  });
}

function tableRow(slide, y, cells, widths, x = 80, rowH = 43, fill = C.white, bold = false) {
  let left = x;
  cells.forEach((cell, i) => {
    shape(slide, { left, top: y, width: widths[i], height: rowH }, fill, C.line);
    textbox(slide, cell, { left: left + 8, top: y + 10, width: widths[i] - 16, height: rowH - 12 }, {
      fontSize: 15,
      bold,
      color: bold ? C.navy : C.grayText,
      alignment: i === 0 ? "left" : "center",
    });
    left += widths[i];
  });
}

async function build() {
  await fs.mkdir(outDir, { recursive: true });
  await fs.mkdir(qaDir, { recursive: true });
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });

  // 1 title
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.white;
    shape(slide, { left: 0, top: 0, width: W, height: 150 }, C.navy, C.navy);
    textbox(slide, "轻量级 AI DPD 算法结题答辩", { left: 78, top: 48, width: 900, height: 72 }, { fontSize: 52, bold: true, color: C.white });
    textbox(slide, "第三届华为无线“声光电磁”硬件创新竞赛 09 赛题", { left: 84, top: 170, width: 780, height: 36 }, { fontSize: 25, color: C.navy });
    textbox(slide, "基于 EA-TCN 的可解释、低复杂度、频域约束 DPD 方案", { left: 84, top: 232, width: 920, height: 44 }, { fontSize: 30, bold: true, color: C.grayText });
    textbox(slide, "重庆邮电大学 通信与信息工程学院 · 2026年7月", { left: 84, top: 310, width: 660, height: 32 }, { fontSize: 20, color: C.gray });
    shape(slide, { left: 84, top: 410, width: 720, height: 118 }, C.lightBlue, C.line);
    textbox(slide, "核心结论", { left: 112, top: 430, width: 160, height: 30 }, { fontSize: 24, bold: true, color: C.navy });
    textbox(slide, "主模型 H15/L6/Input-9/FIR43 在 2284 个等效实乘法器下达到 NMSE -40.15 dB 与 worst ACLR -48.77 dBc。", { left: 112, top: 468, width: 650, height: 46 }, { fontSize: 21, color: C.grayText });
    await addImage(slide, logo, { left: 1048, top: 505, width: 112, height: 112 }, "contain", "CQUPT logo");
    footer(slide, 1);
  }

  // 2 agenda
  {
    const slide = presentation.slides.add();
    header(slide, "答辩结构", "Agenda");
    const items = ["赛题理解与指标约束", "EA-TCN 算法方案", "数学模型与公式", "统一评估与复杂度", "主模型、轻量模型和 786 MAC 专项结果", "工程价值与后续工作"];
    items.forEach((t, i) => {
      const y = 135 + i * 72;
      shape(slide, { left: 120, top: y, width: 48, height: 48 }, i < 3 ? C.lightBlue : C.lightGreen, C.line);
      textbox(slide, String(i + 1).padStart(2, "0"), { left: 120, top: y + 10, width: 48, height: 24 }, { fontSize: 20, bold: true, color: C.navy, alignment: "center" });
      textbox(slide, t, { left: 190, top: y + 8, width: 800, height: 36 }, { fontSize: 27, bold: true, color: C.grayText });
    });
    footer(slide, 2);
  }

  // 3 problem
  {
    const slide = presentation.slides.add();
    header(slide, "功放非线性把 DPD 变成宽带发射链路的必要补偿", "Problem");
    bullet(slide, [
      "PA 同时存在 AM-AM、AM-PM 非线性、短时记忆、热记忆和残余群时延。",
      "线性化不足会恶化主带 EVM/NMSE，并在邻道形成谱再生。",
      "DPD 在数字基带侧学习 PA 逆系统，使功放等效输出接近目标线性响应。",
      "轻量化目标要求算法同时满足频谱指标和硬件乘法器预算。"
    ], 76, 150, 560, 21, C.grayText, 58);
    await addImage(slide, img.dpd, { left: 700, top: 158, width: 430, height: 300 }, "contain", "DPD principle");
    shape(slide, { left: 690, top: 488, width: 445, height: 86 }, C.lightBlue, C.line);
    textbox(slide, "答辩主线", { left: 716, top: 506, width: 150, height: 26 }, { fontSize: 21, bold: true, color: C.navy });
    textbox(slide, "用 EA-TCN 提供可解释的非线性记忆建模，用 FIR 与频域 LS refit 守住邻道泄漏。", { left: 716, top: 538, width: 390, height: 26 }, { fontSize: 18, color: C.grayText });
    footer(slide, 3);
  }

  // 4 task and constraints
  {
    const slide = presentation.slides.add();
    header(slide, "赛题任务需要同时按时域和频域判定", "Task");
    formula(slide, "\\widehat{y}[n] = f_{\\theta}(x)[n],\\quad x[n]=I[n]+jQ[n]", 78, 135, 540, 98, "DPD 直接拟合目标");
    bullet(slide, [
      "输入：复数 IQ 序列 x[n]；输出：pred_y；目标：target_y。",
      "切分：前 80% 训练，后 20% 验证，按时序切分。",
      "采样率：491.52 MHz；主带：[-100,100] MHz。",
      "邻道：[-245,-180] MHz 与 [180,245] MHz。"
    ], 82, 270, 565, 20, C.grayText, 52);
    const x = 690;
    tableRow(slide, 150, ["指标", "门限"], [220, 250], x, 52, C.navy, true);
    tableRow(slide, 202, ["NMSE", "<= -40 dB"], [220, 250], x);
    tableRow(slide, 245, ["worst ACLR", "<= -45 dBc"], [220, 250], x);
    tableRow(slide, 288, ["复杂度", "等效实数乘法器数"], [220, 250], x);
    shape(slide, { left: 690, top: 372, width: 470, height: 124 }, C.lightOrange, "none");
    textbox(slide, "口径说明", { left: 716, top: 393, width: 160, height: 28 }, { fontSize: 23, bold: true, color: C.orange });
    textbox(slide, "本报告所有最终 checkpoint 使用统一评估脚本输出 NMSE、ACLR、PSD 图和乘法器数，避免不同阶段脚本的局部口径差异。", { left: 716, top: 431, width: 410, height: 50 }, { fontSize: 18, color: C.grayText });
    footer(slide, 4);
  }

  // 5 framework
  {
    const slide = presentation.slides.add();
    header(slide, "EA-TCN 方案由四个可解释模块串联", "Design");
    const steps = [
      ["Input-9 特征", "I/Q、前后采样、幅度与高阶幅度特征"],
      ["扩张 TCN 主干", "H15/L6, dilations=[1,2,4,8,16,32]"],
      ["谱注意力门控", "幅度包络驱动通道响应调节"],
      ["复数 FIR refit", "43 taps，频域带权 LS 校正"],
    ];
    steps.forEach((s, i) => {
      const x = 75 + i * 290;
      shape(slide, { left: x, top: 155, width: 235, height: 108 }, i === 3 ? C.lightGreen : C.lightBlue, C.line);
      textbox(slide, s[0], { left: x + 18, top: 178, width: 200, height: 26 }, { fontSize: 23, bold: true, color: i === 3 ? C.green : C.navy });
      textbox(slide, s[1], { left: x + 18, top: 214, width: 200, height: 32 }, { fontSize: 16, color: C.grayText });
      if (i < steps.length - 1) textbox(slide, "→", { left: x + 245, top: 188, width: 40, height: 44 }, { fontSize: 38, bold: true, color: C.navy });
    });
    await addImage(slide, img.eatcn, { left: 180, top: 320, width: 430, height: 240 }, "contain", "EA-TCN architecture");
    bullet(slide, [
      "物理先验降低低阶非线性拟合难度。",
      "扩张卷积用较少层数覆盖长记忆。",
      "FIR refit 对残余群时延和频谱翅膀做低风险校正。"
    ], 690, 340, 470, 20, C.grayText, 56);
    footer(slide, 5);
  }

  // 6 input and prior
  {
    const slide = presentation.slides.add();
    header(slide, "Input-9 把低阶 PA 物理先验前置到特征侧", "Input Features");
    formula(slide, "\\phi_p(x[n]) = x[n]\\cdot |x[n]|^{p-1},\\quad p\\in\\{1,3,5\\}", 80, 138, 560, 102, "复数奇次非线性基函数");
    formula(slide, "\\mathbf{u}[n]=[I,Q,I_{\\pm1},Q_{\\pm1},|x|,|x|^3,|x|^5]", 80, 270, 560, 102, "Input-9 特征向量");
    bullet(slide, [
      "I/Q 与前后采样描述短时记忆。",
      "幅度幂次特征暴露 AM-AM / AM-PM 失真。",
      "线性初始化吸收主要缩放和相位旋转。",
      "TCN 主干不必从零学习所有低阶结构。"
    ], 700, 150, 440, 21, C.grayText, 58);
    footer(slide, 6);
  }

  // 7 tcn + attention
  {
    const slide = presentation.slides.add();
    header(slide, "扩张 TCN 与谱注意力共同处理长记忆和幅度依赖", "TCN + Attention");
    await addImage(slide, img.tcn, { left: 70, top: 135, width: 450, height: 360 }, "contain", "TCN dilation");
    formula(slide, "R \\approx 1+(K-1)\\sum_l d_l = 1+4(1+2+4+8+16+32)=253", 575, 130, 580, 90, "H15/L6 主干感受野");
    formula(slide, "\\mathbf{g}[n]=\\sigma\\left(W_2\\psi(W_1\\mathbf{a}[n])\\right),\\quad \\widetilde{\\mathbf{z}}[n]=\\mathbf{g}[n]\\odot\\mathbf{z}[n]", 575, 255, 580, 112, "谱注意力门控");
    bullet(slide, [
      "扩张因子覆盖到 32 个采样级别。",
      "门控由 |x|、|x|^3、|x|^5 等幅度包络特征驱动。",
      "保持并行卷积推理路径，避免循环状态。"
    ], 585, 410, 540, 20, C.grayText, 52);
    footer(slide, 7);
  }

  // 8 FIR LS
  {
    const slide = presentation.slides.add();
    header(slide, "输出 FIR 与频域带权 LS 直接约束邻道泄漏", "FIR Refit");
    formula(slide, "\\widehat{y}[n]=\\sum_{m=0}^{M-1}c_m z[n-m]", 84, 140, 530, 92, "复数 FIR 输出");
    formula(slide, "\\min_{\\mathbf{c}}\\ \\left\\|\\mathbf{W}\\left(\\mathbf{FZc}-\\mathbf{FY}\\right)\\right\\|_2^2+\\lambda\\|\\mathbf{c}\\|_2^2", 84, 270, 650, 116, "频域带权最小二乘 refit");
    bullet(slide, [
      "FIR 不是增加无约束黑盒容量，而是标准 DSP 线性校正。",
      "固定 TCN 主干后，FIR 系数可稳定求解。",
      "主带、邻道和其他频点可设置不同频域权重。",
      "对 ACLR 的改善比只优化时域 MSE 更直接。"
    ], 790, 150, 400, 20, C.grayText, 56);
    footer(slide, 8);
  }

  // 9 metrics formulas
  {
    const slide = presentation.slides.add();
    header(slide, "指标公式把主带拟合和邻道泄漏分开衡量", "Metrics");
    formula(slide, "\\mathrm{NMSE}=\\frac{\\sum_n|\\widehat{y}[n]-y[n]|^2}{\\sum_n|y[n]|^2}", 78, 128, 560, 105, "验证集 NMSE");
    formula(slide, "\\mathrm{NMSE}_{dB}=10\\log_{10}(\\mathrm{NMSE})", 78, 260, 560, 85, "dB 表达");
    formula(slide, "\\mathrm{ACLR}_{dBc}=\\operatorname{mean}(\\mathrm{PSD}_{adj})-\\operatorname{mean}(\\mathrm{PSD}_{main})", 680, 128, 535, 112, "邻道泄漏比");
    formula(slide, "\\mathrm{ACLR}_{worst}=\\max(\\mathrm{ACLR}_{lower},\\mathrm{ACLR}_{upper})", 680, 275, 535, 92, "上下邻道 worst 口径");
    shape(slide, { left: 240, top: 430, width: 790, height: 88 }, C.lightBlue, "none");
    textbox(slide, "判定逻辑：NMSE 反映主带复数时域拟合，ACLR 反映邻道频谱泄漏；两者必须同时满足才算达标。", { left: 275, top: 457, width: 720, height: 28 }, { fontSize: 21, bold: true, color: C.navy });
    footer(slide, 9);
  }

  // 10 complexity
  {
    const slide = presentation.slides.add();
    header(slide, "复杂度统计把结构压缩转化为乘法器预算", "Complexity");
    formula(slide, "4P+C_{in}H+B(HK+H^2)+2H+3HC_a+4M", 72, 130, 500, 98, "等效实数乘法器数");
    textbox(slide, "统计项：显式复数多项式、输入投影、TCN K-tap 卷积、逐点混合、输出投影、谱注意力和 M-tap 复数 FIR。", { left: 80, top: 258, width: 520, height: 80 }, { fontSize: 20, color: C.grayText });
    const widths = [150, 90, 100, 120, 120];
    tableRow(slide, 132, ["模型", "MAC", "NMSE/dB", "worst ACLR", "结论"], widths, 635, 44, C.navy, true);
    const rows = [
      ["主模型", "2284", "-40.15", "-48.77", "达标"],
      ["最小达标", "2268", "-40.03", "-45.04", "达标"],
      ["998 模型", "998", "-37.37", "-45.30", "ACLR 达标"],
      ["796 模型", "796", "-35.20", "-47.03", "ACLR 达标"],
      ["786 专项", "786", "-35.46", "-46.76", "ACLR 达标"],
    ];
    rows.forEach((r, i) => tableRow(slide, 176 + i * 44, r, widths, 635, 44, i % 2 ? C.white : "#FAFCFE"));
    footer(slide, 10);
  }

  // 11 workflow
  {
    const slide = presentation.slides.add();
    header(slide, "统一评估流程保证不同模型可横向比较", "Evaluation");
    const nodes = [
      ["checkpoint", 100, 150],
      ["验证集推理\\npred_y", 100, 260],
      ["时域误差\\nNMSE / linear NMSE", 430, 260],
      ["频域估计\\nPSD / lower / upper ACLR", 760, 260],
      ["统一输出\\nCSV / 图表 / MAC", 430, 430],
    ];
    nodes.forEach(([txt, x, y], i) => {
      shape(slide, { left: x, top: y, width: 230, height: 72 }, i === 4 ? C.lightGreen : C.lightBlue, C.line);
      textbox(slide, txt.replace("\\n", "\n"), { left: x + 12, top: y + 16, width: 205, height: 44 }, { fontSize: 19, bold: true, color: i === 4 ? C.green : C.navy, alignment: "center" });
    });
    textbox(slide, "→", { left: 250, top: 275, width: 80, height: 44 }, { fontSize: 42, bold: true, color: C.navy, alignment: "center" });
    textbox(slide, "→", { left: 585, top: 275, width: 80, height: 44 }, { fontSize: 42, bold: true, color: C.navy, alignment: "center" });
    textbox(slide, "↘", { left: 500, top: 338, width: 80, height: 44 }, { fontSize: 42, bold: true, color: C.navy, alignment: "center" });
    textbox(slide, "↙", { left: 810, top: 338, width: 80, height: 44 }, { fontSize: 42, bold: true, color: C.navy, alignment: "center" });
    bullet(slide, [
      "统一脚本输出线性 NMSE、dB NMSE、上下邻道 ACLR 和 worst ACLR。",
      "PSD 图用于复核主带谱形和邻道泄漏。",
      "表格数值均以统一评估输出为准。"
    ], 100, 545, 940, 19, C.grayText, 42);
    footer(slide, 11);
  }

  // 12 main result
  {
    const slide = presentation.slides.add();
    header(slide, "主模型达成双指标并保留 ACLR 余量", "Main Result");
    metric(slide, "满足 -40 dB 目标", "NMSE -40.15 dB", "达标", 70, 130, 280, C.green, C.lightGreen);
    metric(slide, "相对门限约 3.77 dB 余量", "worst ACLR -48.77 dBc", "达标", 70, 248, 280, C.green, C.lightGreen);
    shape(slide, { left: 70, top: 374, width: 280, height: 100 }, C.lightBlue, "none");
    textbox(slide, "结构", { left: 92, top: 392, width: 90, height: 24 }, { fontSize: 21, bold: true, color: C.navy });
    textbox(slide, "H15/L6/Input-9/FIR43\n2284 real multipliers", { left: 92, top: 424, width: 230, height: 44 }, { fontSize: 18, color: C.grayText });
    await addImage(slide, img.mainPsd, { left: 400, top: 128, width: 470, height: 330 }, "contain", "Main model PSD ACLR");
    await addImage(slide, img.mainAclr, { left: 900, top: 135, width: 260, height: 155 }, "contain", "Main ACLR bar");
    await addImage(slide, img.mainConv, { left: 900, top: 330, width: 260, height: 155 }, "contain", "Main convergence");
    textbox(slide, "频域校正显著压低邻道谱翅膀，训练后期逐步逼近 -40 dB NMSE 目标。", { left: 405, top: 505, width: 740, height: 34 }, { fontSize: 20, color: C.grayText });
    footer(slide, 12);
  }

  // 13 lightweight
  {
    const slide = presentation.slides.add();
    header(slide, "轻量模型证明 ACLR 可保，NMSE 受主干容量限制", "Lightweight");
    textbox(slide, "998 MAC · Input-9/attention4/FIR39", { left: 95, top: 120, width: 420, height: 28 }, { fontSize: 22, bold: true, color: C.navy });
    await addImage(slide, img.light998, { left: 80, top: 160, width: 480, height: 285 }, "contain", "998 PSD ACLR");
    metric(slide, "ACLR -45.30 dBc", "NMSE -37.37 dB", "ACLR 达标 / NMSE 未达标", 100, 465, 420, C.orange, C.lightOrange);
    textbox(slide, "796 MAC · H10/L3/Input-6/FIR41", { left: 705, top: 120, width: 420, height: 28 }, { fontSize: 22, bold: true, color: C.navy });
    await addImage(slide, img.light796, { left: 690, top: 160, width: 480, height: 285 }, "contain", "796 PSD ACLR");
    metric(slide, "ACLR -47.03 dBc", "NMSE -35.20 dB", "ACLR 达标 / NMSE 未达标", 710, 465, 420, C.orange, C.lightOrange);
    footer(slide, 13);
  }

  // 14 786 special
  {
    const slide = presentation.slides.add();
    header(slide, "786 MAC 专项模型在极低预算下保持 ACLR 达标", "786 MAC");
    shape(slide, { left: 68, top: 120, width: 430, height: 170 }, C.lightBlue, C.line);
    textbox(slide, "结构与 checkpoint", { left: 92, top: 140, width: 240, height: 26 }, { fontSize: 22, bold: true, color: C.navy });
    textbox(slide, "H10/L4/Input-6/pointwise rank 3/FIR41\n786 real multipliers\nstep800_scratch_h10_l4_base6_rank3_fir41_786.pth", { left: 92, top: 178, width: 370, height: 86 }, { fontSize: 16, color: C.grayText });
    metric(slide, "距 -40 dB 约 4.54 dB", "NMSE -35.46 dB", "未达标", 535, 120, 270, C.orange, C.lightOrange);
    metric(slide, "相对门限约 1.76 dB 余量", "worst ACLR -46.76 dBc", "ACLR 达标", 835, 120, 300, C.green, C.lightGreen);
    await addImage(slide, img.model786Psd, { left: 70, top: 335, width: 430, height: 255 }, "contain", "786 PSD ACLR");
    await addImage(slide, img.model786Aclr, { left: 535, top: 330, width: 210, height: 135 }, "contain", "786 ACLR bar");
    await addImage(slide, img.model786Conv, { left: 535, top: 475, width: 210, height: 135 }, "contain", "786 convergence");
    bullet(slide, [
      "FIR41 保住邻道泄漏，ACLR 有约 1.76 dB 门限余量。",
      "L4 + rank3 pointwise 接近 796 baseline，但主干容量不足以达到 -40 dB。",
      "<=800 MAC 的主要瓶颈是低复杂度主干的非线性拟合能力。"
    ], 790, 335, 380, 18, C.grayText, 62);
    footer(slide, 14);
  }

  // 15 engineering value
  {
    const slide = presentation.slides.add();
    header(slide, "TCN + FIR 路线更适合端侧流水部署", "Engineering Value");
    const cards = [
      ["频谱合规", "主模型 worst ACLR -48.77 dBc，相对 -45 dBc 目标有约 3.77 dB 余量。"],
      ["主带精度", "主模型 NMSE -40.15 dB，满足结题目标。"],
      ["可压缩起点", "2268/2284 MAC 达标模型可作为剪枝、低秩和 QAT 的教师或初始化。"],
      ["部署友好", "扩张卷积、逐点混合和 FIR 均可映射为确定性乘加流水。"],
    ];
    cards.forEach((c, i) => {
      const x = i % 2 === 0 ? 90 : 680;
      const y = i < 2 ? 145 : 365;
      shape(slide, { left: x, top: y, width: 500, height: 145 }, i < 2 ? C.lightGreen : C.lightBlue, C.line);
      textbox(slide, c[0], { left: x + 24, top: y + 22, width: 180, height: 30 }, { fontSize: 24, bold: true, color: i < 2 ? C.green : C.navy });
      textbox(slide, c[1], { left: x + 24, top: y + 66, width: 440, height: 54 }, { fontSize: 19, color: C.grayText });
    });
    footer(slide, 15);
  }

  // 16 conclusion
  {
    const slide = presentation.slides.add();
    header(slide, "结题结论聚焦达标交付和低复杂度边界", "Conclusion");
    shape(slide, { left: 78, top: 130, width: 520, height: 150 }, C.lightGreen, "none");
    textbox(slide, "已完成目标", { left: 108, top: 158, width: 200, height: 30 }, { fontSize: 26, bold: true, color: C.green });
    textbox(slide, "EA-TCN 主模型在 2284 MAC 下达到 NMSE -40.15 dB 与 worst ACLR -48.77 dBc，满足赛题双指标。", { left: 108, top: 203, width: 450, height: 54 }, { fontSize: 20, color: C.grayText });
    shape(slide, { left: 78, top: 325, width: 520, height: 150 }, C.lightOrange, "none");
    textbox(slide, "轻量化边界", { left: 108, top: 353, width: 220, height: 30 }, { fontSize: 26, bold: true, color: C.orange });
    textbox(slide, "998、796、786 MAC 模型均能保持 ACLR 达标，但 NMSE 未达标，极低复杂度下主干容量是关键限制。", { left: 108, top: 398, width: 450, height: 54 }, { fontSize: 20, color: C.grayText });
    bullet(slide, [
      "从 2284/2268 MAC 达标模型出发做结构化剪枝。",
      "对 pointwise 混合层继续做低秩分解与恢复训练。",
      "引入 QAT 定点量化，评估 FPGA/DSP 流水实现。",
      "保留频域带权 refit，持续约束邻道泄漏。"
    ], 700, 150, 430, 21, C.grayText, 64);
    textbox(slide, "谢谢各位老师指导", { left: 720, top: 560, width: 380, height: 40 }, { fontSize: 30, bold: true, color: C.navy, alignment: "center" });
    footer(slide, 16);
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const png = await presentation.export({ slide, format: "png", scale: 1 });
    await writeBlob(path.join(qaDir, `slide-${String(index + 1).padStart(2, "0")}.png`), png);
    const layout = await slide.export({ format: "layout" });
    await fs.writeFile(path.join(qaDir, `slide-${String(index + 1).padStart(2, "0")}.layout.json`), await layout.text(), "utf8");
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(qaDir, "montage.webp"), montage);
  const pptx = await PresentationFile.exportPptx(presentation);
  await pptx.save(finalPptx);
  console.log(`Wrote ${finalPptx}`);
}

build().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
