import streamlit as st
import time
import sys
import io


# 拦截终端 print，防止 Baostock 的打印信息搞乱网页前端
class StreamlitRedirect:
    def write(self, text):
        pass

    def flush(self):
        pass


# 设置页面宽屏模式，并起个响亮的名字
st.set_page_config(page_title="Stock Agent AI", page_icon="📈", layout="wide")

st.title("📈 智能多Agent股票投研系统")
st.markdown("基于 LangGraph 与 DeepSeek，融合**技术面、基本面、风控、新闻面及红利策略**的全方位分析。")

# 导入你的核心图逻辑
from run import find_code
from graph import app

# 启动后台维护：日更 panel 数据 / 月更重训模型（不阻塞页面加载、不抛异常）
from tools.auto_maintain import start_maintenance
start_maintenance()

# 布局：左边输入，右边展示
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("分析设置")
    user_input = st.text_input("输入股票代码或名称", placeholder="如：600519 或 中国神华")
    analyze_btn = st.button("🚀 开始深度分析", use_container_width=True)

with col2:
    if analyze_btn:
        if not user_input.strip():
            st.warning("请输入股票代码或名称！")
        else:
            # 1. 查代码阶段
            with st.spinner("正在解析股票代码..."):
                code = find_code(user_input)

            if not code:
                st.error(f"未找到股票：{user_input}，请检查拼写或输入6位代码。")
            else:
                bs_code = code[:2] + "." + code[2:]
                st.info(f"✅ 成功锁定代码：**{bs_code}**")

                # 2. 开始跑大模型图
                st.markdown("### ⏳ Agent 分析中...")
                progress_text = "技术、基本、风控、新闻面 Agent 正在并行工作，请稍候 (约 10-20 秒)..."
                my_bar = st.progress(0, text=progress_text)

                # 搞个小动画模拟进度条（因为 LangGraph 跑起来是黑盒）
                for percent_complete in range(100):
                    time.sleep(0.15)  # 稍微卡一下，让用户看到进度条在动
                    if percent_complete == 99:
                        # 真正的执行逻辑在这里（由于耗时，我们放在最后）
                        # 拦截 print
                        old_stdout = sys.stdout
                        sys.stdout = StreamlitRedirect()

                        try:
                            result = app.invoke({"code": code, "bs_code": bs_code})
                            report = result.get("report", "分析生成失败。")
                        except Exception as e:
                            report = f"系统运行出错：{str(e)}"
                        finally:
                            sys.stdout = old_stdout  # 恢复打印

                    my_bar.progress(percent_complete + 1, text=progress_text)

                my_bar.empty()  # 进度条跑完就隐藏

                # 3. 渲染报告
                st.success("分析完成！")
                st.divider()
                st.markdown(report)
    else:
        # 默认展示的欢迎语
        st.info("👈 请在左侧输入股票代码，然后点击“开始深度分析”。")
        st.markdown("""
        **系统特色：**
        *   **多智能体并行**：拒绝单打独斗，四大分析师同时开工。
        *   **智能分诊路由**：自动识别高股息股票，额外唤醒“红利专项分析师”排查价值陷阱。
        *   **双轨财务提取**：单季环比 + 历史同比，精准识别成长股与价值股。
        """)