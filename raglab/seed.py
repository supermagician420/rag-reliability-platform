"""Small fictional corpus used by the offline demo and tests."""

from typing import Any, Dict, List

from .store import Store


DOCUMENTS = [
    {
        "title": "请假与休假",
        "text": (
            "员工请假需在 OA 系统提交申请，并提前告知直属主管。"
            "病假需提前 1 个工作日申请；如遇急诊，可先电话报备，并在返岗后 3 个工作日内补交医院证明。"
            "年假每年 5 天，未使用不结转，也不折现。"
        ),
    },
    {
        "title": "费用报销",
        "text": (
            "发票需在消费后 30 天内提交，逾期原则上不予报销。"
            "单笔金额超过 5000 元需部门负责人审批。"
            "打车费凭发票报销，并在报销单中注明起止地点和事由。"
        ),
    },
    {
        "title": "IT 报修",
        "text": (
            "电脑、打印机或网络故障，请在企业微信的 IT 报修应用提交工单，"
            "通常 15 分钟内响应。不要自行拆机；如需更换硬件，由 IT 安排。"
            "紧急故障可拨打 IT 值班电话 8888。"
        ),
    },
    {
        "title": "信息安全",
        "text": (
            "密码每 90 天需要更换一次，长度至少 12 位，不得与个人账号相同。"
            "禁止共享账号或将公司资料上传到个人网盘。"
            "使用 U 盘前需经过 IT 加密登记。"
        ),
    },
    {
        "title": "员工福利",
        "text": (
            "公司每年为正式员工安排一次年度体检，入职满 3 个月即可预约。"
            "办公室每周五下午提供下午茶。"
            "补充医疗保险的报销范围以保险公司条款为准。"
        ),
    },
]


def seed_store(store: Store) -> List[Dict[str, Any]]:
    created: List[Dict[str, Any]] = []
    for document in DOCUMENTS:
        created.append(store.add_document(document["title"], document["text"]))
    return created


def seeded_store(db_path: str) -> Store:
    store = Store(db_path)
    if store.count_documents() == 0:
        seed_store(store)
    return store
