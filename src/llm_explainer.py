import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger("llm_explainer")

# Tự động đọc file .env nếu có
def _load_env_file():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k not in os.environ:
                        os.environ[k] = v

_load_env_file()

# Bản đồ giải nghĩa đặc trưng ngân hàng sang ngôn ngữ nghiệp vụ dễ hiểu
FEATURE_BUSINESS_TRANSLATIONS = {
    'name_email_similarity': {
        'high_risk': "Tên khách hàng và địa chỉ email có độ tương đồng rất thấp (nghi ngờ email tạo tự động)",
        'low_risk': "Tên khách hàng và địa chỉ email trùng khớp tự nhiên"
    },
    'current_address_months_count': {
        'high_risk': "Thời gian cư trú tại địa chỉ hiện tại quá ngắn hoặc mới chuyển đến gần đây",
        'low_risk': "Thời gian cư trú tại địa chỉ hiện tại ổn định, lâu năm"
    },
    'prev_address_months_count': {
        'high_risk': "Lịch sử địa chỉ cư trú trước đây không rõ ràng hoặc có gián đoạn",
        'low_risk': "Lịch sử cư trú trước đây rõ ràng"
    },
    'income': {
        'high_risk': "Mức thu nhập khai báo ở mức rủi ro cao hoặc không tương xứng",
        'low_risk': "Mức thu nhập ổn định và phù hợp"
    },
    'customer_age': {
        'high_risk': "Độ tuổi khách hàng nằm trong nhóm có tỷ lệ gian lận cao trong dữ liệu lịch sử",
        'low_risk': "Độ tuổi thuộc nhóm nhân khẩu học ổn định"
    },
    'days_since_request': {
        'high_risk': "Khoảng thời gian từ khi phát sinh yêu cầu đến khi hoàn tất hồ sơ bất thường",
        'low_risk': "Thời gian xử lý yêu cầu diễn ra bình thường"
    },
    'intended_balcon_amount': {
        'high_risk': "Số dư chuyển khoản dự kiến cao bất thường hoặc có dấu hiệu rút sạch hạn mức",
        'low_risk': "Số dư dự kiến giao dịch ở mức an toàn"
    },
    'zip_count_4w': {
        'high_risk': "Phát hiện nhiều hồ sơ mở tài khoản từ cùng một mã bưu chính trong 4 tuần qua",
        'low_risk': "Mật độ đăng ký tại khu vực bưu chính ở mức bình thường"
    },
    'velocity_6h': {
        'high_risk': "Tần suất đăng ký/giao dịch dồn dập trong 6 giờ qua tăng đột biến",
        'low_risk': "Tần suất thao tác trong 6 giờ qua bình thường"
    },
    'velocity_24h': {
        'high_risk': "Số lượng thao tác trong 24 giờ qua vượt ngưỡng an toàn",
        'low_risk': "Tần suất giao dịch trong 24 giờ qua ổn định"
    },
    'velocity_4w': {
        'high_risk': "Tổng số lượt mở thẻ/giao dịch trong 4 tuần qua cao bất thường",
        'low_risk': "Lịch sử thao tác 4 tuần qua bình thường"
    },
    'bank_branch_count_8w': {
        'high_risk': "Có nhiều giao dịch tại các chi nhánh khác nhau trong 8 tuần qua",
        'low_risk': "Địa điểm giao dịch tập trung tại chi nhánh quen thuộc"
    },
    'date_of_birth_distinct_emails_4w': {
        'high_risk': "Ngày sinh của hồ sơ này được sử dụng với nhiều email khác nhau trong 4 tuần qua (dấu hiệu giả mạo danh tính)",
        'low_risk': "Thông tin ngày sinh duy nhất và nhất quán"
    },
    'credit_risk_score': {
        'high_risk': "Điểm rủi ro tín dụng nội bộ ở mức cảnh báo cao",
        'low_risk': "Điểm tín dụng nội bộ tốt"
    },
    'bank_months_count': {
        'high_risk': "Thời gian thiết lập quan hệ với ngân hàng quá ngắn (khách hàng mới toanh)",
        'low_risk': "Khách hàng lâu năm có quan hệ tín dụng gắn bó"
    },
    'proposed_credit_limit': {
        'high_risk': "Hạn mức tín dụng đề xuất cao bất thường so với hồ sơ",
        'low_risk': "Hạn mức đề xuất hợp lý"
    },
    'session_length_in_minutes': {
        'high_risk': "Thời lượng phiên truy cập quá ngắn (nghi ngờ script/bot tự động điền form)",
        'low_risk': "Thời lượng thao tác tương tác người dùng tự nhiên"
    },
    'device_distinct_emails_8w': {
        'high_risk': "Thiết bị này đã từng liên kết với nhiều tài khoản email khác nhau trong 8 tuần qua (dấu hiệu nông trại tài khoản / botnet)",
        'low_risk': "Thiết bị chỉ liên kết với duy nhất một tài khoản chính chủ"
    },
    'device_fraud_count': {
        'high_risk': "Thiết bị này đã từng bị ghi nhận gian lận trong quá khứ (Blacklisted Device)",
        'low_risk': "Thiết bị trong sạch, chưa từng vi phạm"
    },
    'email_is_free': {
        'high_risk': "Sử dụng dịch vụ email miễn phí / dùng một lần thay vì email doanh nghiệp/cá nhân uy tín",
        'low_risk': "Tên miền email có độ tin cậy"
    },
    'foreign_request': {
        'high_risk': "Yêu cầu đăng ký được gửi từ địa chỉ IP quốc tế / VPN chuyển vùng",
        'low_risk': "Địa chỉ IP nội địa hợp lệ"
    },
    'has_other_cards': {
        'high_risk': "Hồ sơ chưa có lịch sử sở hữu thẻ tín dụng tại các tổ chức khác",
        'low_risk': "Đã có thẻ tín dụng tại tổ chức khác"
    },
    'keep_alive_session': {
        'high_risk': "Phiên kết nối sử dụng tín hiệu duy trì bất thường",
        'low_risk': "Tương tác phiên tự nhiên"
    },
    'housing_status': {
        'high_risk': "Tình trạng cư trú thuộc nhóm rủi ro cao (nhà thuê ngắn hạn/không cố định)",
        'low_risk': "Tình trạng sở hữu nhà ở ổn định"
    },
    'employment_status': {
        'high_risk': "Tình trạng việc làm không ổn định hoặc tự do không chứng minh được dòng tiền",
        'low_risk': "Công việc ổn định có hợp đồng"
    }
}


def rule_based_explainer_vi(
    risk_score: float,
    threshold: float,
    top_reasons: List[Dict[str, Any]],
    customer_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Sinh lời giải thích tiếng Việt tự nhiên chuẩn xác theo thuật toán quy tắc nghiệp vụ (Domain Expert NLG).
    Đảm bảo luôn chạy 100% offline không phụ thuộc mạng hay API key.
    """
    is_fraud = risk_score >= threshold
    percent_score = risk_score * 100
    
    if is_fraud:
        header = f"🚨 **CẢNH BÁO GIAN LẬN (Điểm rủi ro: {percent_score:.1f}% - Vượt ngưỡng {threshold*100:.1f}%)**\n\n"
        header += "Hệ thống AI đề xuất **CHẶN / CHUYỂN REVIEW THỦ CÔNG** do phát hiện các dấu hiệu bất thường sau:\n"
    else:
        header = f"✅ **GIAO DỊCH HỢP LỆ (Điểm rủi ro: {percent_score:.1f}% - Dưới ngưỡng {threshold*100:.1f}%)**\n\n"
        header += "Hệ thống AI đề xuất **TỰ ĐỘNG PHÊ DUYỆT**. Các yếu tố đánh giá an toàn:\n"
        
    reason_lines = []
    idx = 1
    for item in top_reasons:
        f_name = item.get('feature', '')
        f_val = item.get('value', '')
        s_val = item.get('shap_value', 0.0)
        
        trans_dict = FEATURE_BUSINESS_TRANSLATIONS.get(f_name, None)
        if trans_dict:
            desc = trans_dict['high_risk'] if s_val > 0 else trans_dict['low_risk']
        else:
            impact_desc = "tăng nguy cơ rủi ro" if s_val > 0 else "giảm nguy cơ rủi ro"
            desc = f"Đặc trưng `{f_name}` (giá trị: {f_val}) đóng góp {impact_desc}"
            
        sign = "+" if s_val > 0 else ""
        reason_lines.append(f"{idx}. **{desc}** *(Đóng góp SHAP: {sign}{s_val:.3f})*")
        idx += 1
        
    summary_action = (
        "\n\n📋 **Khuyến nghị cho Compliance Officer**: "
        + ("Yêu cầu khách hàng xác thực sinh trắc học bổ sung (eKYC) hoặc liên hệ đối chiếu nguồn thu nhập trước khi mở tài khoản." 
           if is_fraud else "Hồ sơ đủ điều kiện giải ngân/cấp hạn mức tự động.")
    )
    
    return header + "\n".join(reason_lines) + summary_action


def openai_llm_explainer_vi(
    risk_score: float,
    threshold: float,
    top_reasons: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Gọi mô hình OpenAI (GPT-4o-mini / GPT-4o) để tạo báo cáo giải trình nghiệp vụ chuyên nghiệp.
    """
    import urllib.request
    
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Không tìm thấy OPENAI_API_KEY.")
        
    is_fraud = risk_score >= threshold
    prompt = f"""
Bạn là Trưởng phòng Thẩm định Rủi ro & Chống Gian lận (Senior Fraud Risk Lead) tại một Ngân hàng Thương mại lớn.
Mô hình Machine Learning (XGBoost) vừa chấm điểm một hồ sơ mở tài khoản với kết quả kỹ thuật như sau:

- Điểm rủi ro gian lận (Fraud Risk Score): {risk_score*100:.2f}%
- Ngưỡng quyết định rủi ro (Risk Threshold): {threshold*100:.2f}%
- Quyết định AI: {'🚨 CẢNH BÁO GIAN LẬN (CHẶN / REVIEW THỦ CÔNG)' if is_fraud else '✅ PHÊ DUYỆT TỰ ĐỘNG'}

Top 5 đặc trưng có đóng góp SHAP lớn nhất (SHAP > 0 là tăng nguy cơ gian lận, SHAP < 0 là bảo vệ an toàn):
{json.dumps(top_reasons, ensure_ascii=False, indent=2)}

Hãy viết một bản tóm tắt giải trình nghiệp vụ súc tích, chuyên nghiệp bằng tiếng Việt (khoảng 3 đoạn có icon và định dạng markdown rõ ràng) dành cho nhân viên phòng Compliance / Điều tra gian lận:
1. Kết luận quyết định và mức độ rủi ro tổng quan.
2. Phân tích chi tiết top lý do cốt lõi (chuyển đổi các đặc trưng kỹ thuật thành ngôn ngữ ngân hàng như độ tương đồng email/tên, hành vi thiết bị, lịch sử cư trú, thu nhập).
3. Đưa ra khuyến nghị hành động cụ thể tiếp theo cho chuyên viên xử lý hồ sơ.
"""

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Bạn là chuyên gia thẩm định gian lận ngân hàng chuyên nghiệp và sắc sảo."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 500
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers
    )
    
    with urllib.request.urlopen(req, timeout=12) as response:
        res_json = json.loads(response.read().decode('utf-8'))
        text = res_json['choices'][0]['message']['content']
        return text


def gemini_llm_explainer_vi(
    risk_score: float,
    threshold: float,
    top_reasons: List[Dict[str, Any]],
    api_key: Optional[str] = None
) -> str:
    """
    Gọi mô hình Google Gemini để tạo đoạn văn giải thích nghiệp vụ.
    """
    import urllib.request
    
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("Không tìm thấy GEMINI_API_KEY.")
        
    prompt = f"""
Bạn là chuyên gia thẩm định rủi ro và phòng chống gian lận tài chính (Senior Fraud & Risk Analyst) tại một Ngân hàng lớn.
Mô hình Machine Learning (XGBoost) vừa chấm điểm một hồ sơ mở tài khoản với kết quả như sau:

- Điểm rủi ro gian lận (Fraud Risk Score): {risk_score*100:.2f}%
- Ngưỡng quyết định rủi ro (Risk Threshold): {threshold*100:.2f}%
- Quyết định: {'CHẶN VÀ ĐƯA VÀO REVIEW THỦ CÔNG' if risk_score >= threshold else 'PHÊ DUYỆT TỰ ĐỘNG'}

Top các đặc trưng có đóng góp SHAP lớn nhất:
{json.dumps(top_reasons, ensure_ascii=False, indent=2)}

Hãy viết một bản tóm tắt giải trình nghiệp vụ súc tích, chuyên nghiệp bằng tiếng Việt dành cho nhân viên phòng Compliance.
"""
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
    req_body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(req_body).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    with urllib.request.urlopen(req, timeout=10) as response:
        res_json = json.loads(response.read().decode('utf-8'))
        text = res_json['candidates'][0]['content']['parts'][0]['text']
        return text


def generate_fraud_explanation(
    risk_score: float,
    threshold: float = 0.48,
    top_reasons: Optional[List[Dict[str, Any]]] = None,
    customer_context: Optional[Dict[str, Any]] = None,
    use_llm: bool = True
) -> str:
    """
    Hàm tổng hợp để sinh lời giải thích tiếng Việt.
    1. Ưu tiên gọi OpenAI LLM (nếu có OPENAI_API_KEY)
    2. Fallback sang Google Gemini (nếu có GEMINI_API_KEY)
    3. Tự động fallback về Rule-Based Expert NLG nếu không có mạng / key.
    """
    if top_reasons is None:
        top_reasons = []
        
    if use_llm:
        # 1. Thử OpenAI API
        if os.getenv("OPENAI_API_KEY"):
            try:
                return openai_llm_explainer_vi(risk_score, threshold, top_reasons)
            except Exception as e:
                logger.debug(f"OpenAI API unavailable ({e}), falling back to next provider...")
                
        # 2. Thử Gemini API
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            try:
                return gemini_llm_explainer_vi(risk_score, threshold, top_reasons)
            except Exception as e:
                logger.debug(f"Gemini API unavailable ({e}), falling back to rule-based...")

    # 3. Fallback chắc chắn về Rule-based NLG
    return rule_based_explainer_vi(risk_score, threshold, top_reasons, customer_context)
