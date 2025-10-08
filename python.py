# python_dcf_app.py

import streamlit as st
import pandas as pd
import numpy as np
from google import genai
from google.genai.errors import APIError
import io

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Đánh Giá Phương Án Kinh Doanh (DCF) 📈",
    layout="wide"
)

st.title("Ứng dụng Đánh Giá Phương Án Kinh Doanh (DCF) 📊")
st.subheader("Trích xuất thông tin, Xây dựng Dòng tiền và Tính toán Chỉ số Hiệu quả")

# --- Hàm gọi API Gemini để Trích xuất Dữ liệu (Yêu cầu 1) ---
@st.cache_data(show_spinner="Đang gửi file lên Gemini AI để trích xuất dữ liệu...")
def extract_financial_data_with_ai(uploaded_file, api_key):
    """Sử dụng Gemini AI để đọc file và trích xuất các chỉ số tài chính quan trọng."""
    
    # 1. Chuẩn bị file và Client
    client = genai.Client(api_key=api_key)
    model_name = 'gemini-2.5-flash'
    
    # Upload file lên Gemini API
    # Streamlit file uploader returns a BytesIO-like object, which works with genai.Client.files.upload
    file_to_upload = client.files.upload(file=uploaded_file, mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document") 
    
    st.info(f"File **{uploaded_file.name}** đã được tải thành công lên Gemini để phân tích.")

    # 2. Xây dựng Prompt
    prompt = f"""
    Bạn là một chuyên gia phân tích tài chính. Nhiệm vụ của bạn là đọc toàn bộ nội dung trong file Word đính kèm.
    Sau đó, hãy trích xuất **chính xác** các thông tin tài chính sau của dự án, và trả về kết quả **CHỈ DƯỚI DẠNG MỘT JSON OBJECT** (KHÔNG có bất kỳ văn bản giải thích hoặc markdown nào khác ngoài JSON):
    
    1. Vốn đầu tư ban đầu (Initial Investment - Cần là một số, nếu có nhiều phần hãy cộng tổng): 'vốn_đầu_tư'
    2. Dòng đời dự án (Project Life - Số năm): 'dòng_đời_dự_án'
    3. Doanh thu hàng năm (Annual Revenue - Cần là một số ước tính trung bình hoặc năm đầu tiên): 'doanh_thu_hàng_năm'
    4. Chi phí hoạt động hàng năm (Annual Operating Cost - KHÔNG bao gồm Chi phí khấu hao và Lãi vay): 'chi_phí_hàng_năm'
    5. Tỷ lệ chiết khấu WACC (Weighted Average Cost of Capital - Dạng thập phân, ví dụ 10% là 0.1): 'wacc'
    6. Tỷ suất thuế thu nhập doanh nghiệp (Corporate Tax Rate - Dạng thập phân, ví dụ 20% là 0.2): 'thuế'
    7. Khấu hao hàng năm (Annual Depreciation - Cần là một số, nếu không có hãy ước tính dựa trên Vốn đầu tư và Dòng đời dự án theo phương pháp đường thẳng): 'khấu_hao_hàng_năm'

    Nếu một thông tin bị thiếu, hãy sử dụng giá trị ước tính hợp lý và ghi chú rõ ràng trong phần giải thích sau khi hoàn thành JSON.
    
    Ví dụ định dạng JSON bắt buộc:
    {{
        "vốn_đầu_tư": 1000000000,
        "dòng_đời_dự_án": 5,
        "doanh_thu_hàng_năm": 300000000,
        ...
    }}
    
    Dựa trên file: {file_to_upload.name}
    """
    
    # 3. Gọi API và Phân tích JSON
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, file_to_upload],
        )
        
        # Xử lý kết quả trả về: Tìm JSON object
        json_string = response.text.strip()
        # Đảm bảo chỉ lấy phần JSON nếu có ký tự thừa
        if json_string.startswith("```json"):
            json_string = json_string.replace("```json", "").replace("```", "").strip()

        import json
        financial_data = json.loads(json_string)
        
        # Dọn dẹp file đã upload sau khi dùng
        client.files.delete(name=file_to_upload.name)
        
        return financial_data, response.text # Trả về cả dữ liệu đã lọc và phản hồi gốc

    except APIError as e:
        if 'Unsupported file type' in str(e):
             st.error("Lỗi: Gemini API không hỗ trợ loại file này. Vui lòng đảm bảo file là định dạng **.docx**.")
             return None, None
        return st.error(f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"), None
    except json.JSONDecodeError:
        st.error(f"Lỗi phân tích JSON từ AI. Dữ liệu thô từ AI: {response.text}")
        return None, None
    except Exception as e:
        return st.error(f"Đã xảy ra lỗi không xác định trong quá trình trích xuất: {e}"), None


# --- Hàm Tính toán Dòng tiền (Yêu cầu 2) ---
def calculate_cash_flow(data):
    """Xây dựng bảng dòng tiền và tính toán các chỉ số DCF."""
    
    # Chuẩn bị dữ liệu
    N = int(data['dòng_đời_dự_án'])
    I = data['vốn_đầu_tư']
    Rev = data['doanh_thu_hàng_năm']
    Cost = data['chi_phí_hàng_năm']
    WACC = data['wacc']
    Tax = data['thuế']
    Depr = data['khấu_hao_hàng_năm']

    # Xây dựng DataFrame Dòng tiền
    years = [f"Năm {i}" for i in range(1, N + 1)]
    df_cf = pd.DataFrame(index=years)
    
    # Dòng tiền hoạt động (Operating Cash Flow - OCF)
    EBIT = Rev - Cost - Depr
    Tax_Amount = EBIT * Tax if EBIT > 0 else 0
    EAT = EBIT - Tax_Amount
    
    OCF = EAT + Depr # Lợi nhuận sau thuế + Khấu hao
    
    # Thêm dòng tiền vào DataFrame
    df_cf.loc[:, 'Doanh thu (A)'] = Rev
    df_cf.loc[:, 'Chi phí hoạt động (B)'] = Cost
    df_cf.loc[:, 'Khấu hao (C)'] = Depr
    df_cf.loc[:, 'Lợi nhuận trước thuế & lãi (EBIT = A-B-C)'] = EBIT
    df_cf.loc[:, 'Thuế TNDN (D)'] = Tax_Amount
    df_cf.loc[:, 'Lợi nhuận sau thuế (EAT = EBIT - D)'] = EAT
    df_cf.loc[:, 'Dòng tiền Hoạt động (OCF = EAT + C)'] = OCF
    df_cf.loc[:, 'Dòng tiền Thuần (Net Cash Flow)'] = OCF 
    
    # Bổ sung năm 0 (Initial Investment)
    initial_cf = pd.Series([-I], index=['Dòng tiền Thuần'])
    
    # Dòng tiền thuần của dự án
    cash_flows = np.concatenate(([initial_cf['Dòng tiền Thuần']], df_cf['Dòng tiền Thuần'].values))
    
    # Bảng dòng tiền hiển thị
    df_cf_display = df_cf.T
    df_cf_display.insert(0, 'Năm 0', np.nan)
    df_cf_display.loc['Dòng tiền Thuần', 'Năm 0'] = initial_cf['Dòng tiền Thuần']
    df_cf_display = df_cf_display.fillna(0).astype(object) # Chuyển về object để format

    # --- Tính toán Chỉ số (Yêu cầu 3) ---
    
    # 1. NPV (Net Present Value)
    npv = np.npv(WACC, cash_flows)
    
    # 2. IRR (Internal Rate of Return)
    try:
        irr = np.irr(cash_flows)
    except ValueError:
        irr = np.nan # Thường xảy ra nếu dòng tiền không đổi dấu
    
    # 3. PP (Payback Period - Thời gian hoàn vốn)
    cumulative_cf = np.cumsum(cash_flows)
    payback_period = np.where(cumulative_cf >= 0)[0]
    if len(payback_period) > 0:
        period = payback_period[0] - 1
        fraction = -cumulative_cf[period] / cash_flows[period + 1]
        pp = period + fraction
    else:
        pp = np.nan
        
    # 4. DPP (Discounted Payback Period - Thời gian hoàn vốn có chiết khấu)
    discounted_cash_flows = cash_flows / (1 + WACC)**np.arange(len(cash_flows))
    cumulative_dcf = np.cumsum(discounted_cash_flows)
    dpp_period = np.where(cumulative_dcf >= 0)[0]
    if len(dpp_period) > 0:
        period_dpp = dpp_period[0] - 1
        fraction_dpp = -cumulative_dcf[period_dpp] / discounted_cash_flows[period_dpp + 1]
        dpp = period_dpp + fraction_dpp
    else:
        dpp = np.nan

    metrics = {
        'NPV': npv,
        'IRR': irr,
        'PP': pp,
        'DPP': dpp,
        'Dòng đời dự án': N
    }
    
    return df_cf_display, metrics

# --- Hàm gọi API Gemini để Phân tích Chỉ số (Yêu cầu 4) ---
def get_ai_analysis(metrics, raw_data_json, api_key):
    """Gửi các chỉ số hiệu quả dự án đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash'
        
        prompt = f"""
        Bạn là một chuyên gia đánh giá dự án đầu tư. Dựa trên các chỉ số hiệu quả tài chính sau của dự án, hãy đưa ra một đánh giá chuyên sâu và khách quan (khoảng 3-4 đoạn) về khả năng chấp nhận và độ hấp dẫn của dự án:

        1.  **Chỉ số đã tính:**
            - NPV (Giá trị hiện tại ròng): {metrics.get('NPV', 'N/A'):,.0f}
            - IRR (Tỷ suất sinh lời nội bộ): {metrics.get('IRR', 'N/A'):.2%}
            - WACC (Tỷ lệ chiết khấu - Tỷ suất sinh lời yêu cầu): {raw_data_json.get('wacc', 'N/A'):.2%}
            - PP (Thời gian hoàn vốn): {metrics.get('PP', 'N/A'):.2f} năm
            - DPP (Thời gian hoàn vốn có chiết khấu): {metrics.get('DPP', 'N/A'):.2f} năm
            - Dòng đời dự án: {metrics.get('Dòng đời dự án', 'N/A')} năm

        2.  **Lưu ý khi phân tích:**
            - **NPV:** Dự án có nên được chấp nhận hay không (NPV > 0)?
            - **IRR:** So sánh IRR với WACC (IRR > WACC?) để đánh giá khả năng sinh lời.
            - **PP/DPP:** Đánh giá rủi ro và tính thanh khoản của vốn đầu tư so với dòng đời dự án.

        Dữ liệu đầu vào thô đã trích xuất từ file Word:
        {raw_data_json}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- MAIN APP LOGIC ---

# Lấy API Key từ Streamlit Secrets
api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("🔑 **Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'.** Vui lòng cấu hình Khóa trong Streamlit Secrets để sử dụng chức năng AI.")

uploaded_file = st.file_uploader(
    "1. Tải file Word (.docx) chứa Phương án Kinh doanh:",
    type=['docx']
)

# Khởi tạo state để lưu dữ liệu đã lọc
if 'financial_data' not in st.session_state:
    st.session_state.financial_data = None
    st.session_state.raw_ai_output = None

# --- Yêu cầu 1: Lọc dữ liệu ---
if uploaded_file is not None and api_key:
    if st.button("▶️ 1. Lọc Thông tin Dự án bằng AI", type="primary"):
        with st.spinner('Đang phân tích file Word bằng Gemini AI...'):
            data, raw_output = extract_financial_data_with_ai(uploaded_file, api_key)
            if data:
                st.session_state.financial_data = data
                st.session_state.raw_ai_output = raw_output
                st.success("✅ Trích xuất dữ liệu thành công!")
                
if st.session_state.financial_data:
    
    data = st.session_state.financial_data
    
    st.divider()

    ## 2. Dữ liệu Đầu vào Đã Lọc
    
    st.subheader("2. Dữ liệu Đầu vào Đã Lọc (Từ AI) 🔍")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Vốn đầu tư (I)", f"{data.get('vốn_đầu_tư', 0):,.0f} VND")
        st.metric("📈 Doanh thu hàng năm", f"{data.get('doanh_thu_hàng_năm', 0):,.0f} VND")
        st.metric("⏳ Dòng đời dự án (N)", f"{data.get('dòng_đời_dự_án', 0)} năm")

    with col2:
        st.metric("💸 Chi phí hoạt động", f"{data.get('chi_phí_hàng_năm', 0):,.0f} VND")
        st.metric("🛡️ Khấu hao hàng năm", f"{data.get('khấu_hao_hàng_năm', 0):,.0f} VND")
        st.metric("📉 WACC (Tỷ lệ chiết khấu)", f"{data.get('wacc', 0):.2%}")

    with col3:
        st.metric("🧾 Thuế TNDN", f"{data.get('thuế', 0):.2%}")
        
    st.expander("Xem Dữ liệu thô và JSON AI đã trả về").code(st.session_state.raw_ai_output, language='json')
    
    # --- Yêu cầu 2 & 3: Bảng dòng tiền và Tính toán Chỉ số ---
    try:
        df_cf_display, metrics = calculate_cash_flow(data)

        st.divider()

        ## 3. Bảng Dòng tiền (Cash Flow)
        st.subheader("3. Bảng Dòng tiền của Dự án (Cash Flow) 💸")
        
        # Định dạng hiển thị
        st.dataframe(df_cf_display.style.format('{:,.0f}'), use_container_width=True)
        
        st.divider()

        ## 4. Các Chỉ số Đánh giá Hiệu quả
        st.subheader("4. Các Chỉ số Đánh giá Hiệu quả Dự án (NPV, IRR, PP, DPP) 🎯")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            # NPV
            npv_color = "inverse" if metrics['NPV'] < 0 else "off"
            st.metric(
                label="Giá trị hiện tại ròng (NPV)",
                value=f"{metrics['NPV']:,.0f} VND",
                delta="Dự án KHÔNG hấp dẫn" if metrics['NPV'] < 0 else "Dự án HẤP DẪN (NPV > 0)",
                delta_color=npv_color
            )
        with col_m2:
            # IRR
            irr_color = "inverse" if metrics['IRR'] < data['wacc'] else "off"
            st.metric(
                label="Tỷ suất sinh lời nội bộ (IRR)",
                value=f"{metrics['IRR']:.2%}",
                delta=f"WACC: {data['wacc']:.2%}",
                delta_color=irr_color
            )
        with col_m3:
            # PP
            st.metric(
                label="Thời gian Hoàn vốn (PP)",
                value=f"{metrics['PP']:.2f} năm",
                delta=f"Dòng đời: {data['dòng_đời_dự_án']} năm"
            )
        with col_m4:
            # DPP
            st.metric(
                label="Hoàn vốn có Chiết khấu (DPP)",
                value=f"{metrics['DPP']:.2f} năm",
                delta=f"So sánh với PP: {metrics['DPP'] - metrics['PP']:.2f} năm"
            )

        st.divider()

        # --- Yêu cầu 4: Phân tích AI ---
        ## 5. Phân tích Chuyên sâu bằng AI
        st.subheader("5. Phân tích Chuyên sâu về Hiệu quả Dự án (AI) 🧠")
        
        if st.button("✨ Yêu cầu AI Phân tích Hiệu quả Dự án", key='analyze_metrics'):
            if api_key:
                with st.spinner('Đang gửi chỉ số và chờ Gemini AI phân tích...'):
                    ai_result = get_ai_analysis(metrics, data, api_key)
                    st.session_state.ai_analysis_result = ai_result
                    
        if st.session_state.get('ai_analysis_result'):
            st.markdown("**Kết quả Phân tích từ Gemini AI:**")
            st.info(st.session_state.ai_analysis_result)

    except KeyError as e:
        st.error(f"Lỗi: Thiếu dữ liệu quan trọng để tính toán. Kiểm tra lại dữ liệu AI đã lọc (thiếu key: {e}).")
    except ValueError as e:
        st.error(f"Lỗi tính toán: {e}. Vui lòng kiểm tra lại tính hợp lý của các con số đã lọc (ví dụ: dòng đời dự án, WACC).")

else:
    st.info("Vui lòng tải file Word (.docx) và nhấn nút Lọc để bắt đầu phân tích.")

# Xóa state sau khi hoàn thành để chuẩn bị cho lần tải file tiếp theo
if uploaded_file is None:
    st.session_state.financial_data = None
    st.session_state.raw_ai_output = None
    if 'ai_analysis_result' in st.session_state:
        del st.session_state.ai_analysis_result
