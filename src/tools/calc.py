"""
calc_severance 工具：按《劳动合同法》第47条第一款算经济补偿
第47条还有第二款（高薪封顶、十二年上限），这里没实现。
在返回值里写明，模型就会转述给用户，不会假装算全了。
"""
from langchain_core.tools import tool


def make_calc_severance():
  """
  造一个 calc_severance 工具
  """
  @tool 
  def calc_severance(years: float, monthly_salary: float)->str:
    """
    按《劳动合同法》第47条第一款计算经济补偿金（N）。
       
    规则：每满一年支付一个月工资；六个月以上不满一年的按一年计算；不满六个月的支付半个月。
    只在用户明确问了金额、且已知工作年限和月工资时调用。
    缺参数就反问用户，一定不要猜。
    注意：本工具没有考虑特殊情况，如《劳动合同法》第47条第二款的规定。（高薪封顶和年限）
    """
       
    #年限
    if years<0:
      return "工作年限不能为负数，请检查输入。"
    if years<0.5:
      months=0.5   #不满半年则算半个月
    elif years<1:
      months=1     #六个月以上（含）不满一年  按一年
    else:
      whole_years=int(years) #取整
      fraction=years-whole_years  #得到年份差值
      if fraction>=0.5:
        months=float(whole_years+1)   #满半年加一个月
      elif fraction>0:
        months=whole_years+0.5  #不满半年加半个月
      else:
        months=float(whole_years)   #整年
       
    amount=months*monthly_salary    #计算金额
    return(
      f"计算结果：\n"
      f"工作年限 {years} 年 -> 折算 {months} 个月工资\n"
      f"月工资 {monthly_salary:.0f} 元\n"
      f"经济补偿金 = {months} × {monthly_salary:.0f} = {amount:.0f} 元\n"
      f"依据：《中华人民共和国劳动合同法》第四十七条\n"
      f"限制：未计入第二款的高薪封顶与十二年上限"
    )

  return calc_severance