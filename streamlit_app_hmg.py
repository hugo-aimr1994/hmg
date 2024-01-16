import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO
from pyxlsb import open_workbook as open_xlsb
import itertools

#mpl.font_manager.fontManager.addfont('./SimHei.ttf') #临时注册新的全局字体
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 18  #设置字体大小，全局有效

#设置页面标题
st.title("滑摩功计算")
#添加文件上传功能
uploaded_datafile = st.file_uploader("🟦上传原始数据文件",type=["xlsx","csv"])

max_trq = st.number_input('✅最大扭矩',key = '1')
st.write('🟦The current number is ', max_trq)

gearbox_list = {
'DT1425 OD': [13.16,10.54,8.92,7.15,5.74,4.6,3.75,3,2.38,1.91,1.53,1.23,1,0.8],
'DT1432 OD':[12.96,10.35,8.79,7.02,5.65,4.51,3.75,2.99,2.34,1.87,1.51,1.20,1.00,0.80],
'DT1425/28 DD': [16.41,13.16,11.13,8.92,7.16,5.74,4.68,3.75,2.97,2.38,1.91,1.53,1.25,1.00],
'DT1422 OD' :[13.513,10.827,8.922,7.149,5.739,4.598,3.750,3.005,2.379,1.906,1.530,1.226,1.000,0.801],
'DT1422 DD':[16.86,13.51,11.13,8.92,7.16,5.74,4.68,3.75,2.97,2.38,1.91,1.53,1.25,1],
'DA1223 DD': [14.94,11.73,9.04,7.09,5.54,4.35,3.44,2.70,2.08,1.63,1.27,1],
'DA1223 OD': [11.73,9.21,7.09,5.57,4.35,3.41,2.70,2.12,1.63,1.28,1],
'12JS OD' :[12.10,9.52,7.31,5.71,4.46,3.48,2.71,2.11,1.64,1.28,1.00,0.78],
'12JSD180T DD': [15.53,12.08,9.39,7.33,5.73,4.46,3.48,2.71,2.10,1.64,1.28,1.00],
'8JS85E-C OD': [9.65,6.46,4.32,3.19,2.23,1.5,1,0.74]    
}

option = st.selectbox('✅变速箱型号',
('DT1425 OD',
'DT1432 OD',
'DT1425/28 DD',
'DT1422 OD' ,
'DT1422 DD',
'DA1223 DD',
'DA1223 OD',
'12JS OD' ,
'12JSD180T DD',
'8JS85E-C OD'))
gear_list = gearbox_list[option]
st.write('🟦变速箱速比:', gearbox_list[option])

def get_synovial_power(T1,T2,eng_spd1,eng_spd2,input_spd1,input_spd2):#[a,b]之间插入99个点
    T_new = np.linspace(T1,T2,100,endpoint = False)
    eng_spd_new = np.linspace(eng_spd1,eng_spd2, 100,endpoint = False)
    input_spd_new = np.linspace(input_spd1,input_spd2, 100,endpoint = False)
    synovial_power = sum(T_new*abs(eng_spd_new-input_spd_new)*1/100*max_trq/100/9549)
    return synovial_power

def get_speed_ratio_define(speed_ratio_cal):
    speed_ratio_list = gear_list
    speed_ratio_list = sorted(speed_ratio_list, reverse=True)
    speed_ratio_list_mean = []
    for i in range(len(speed_ratio_list) - 1):
        speed_ratio_list_mean.append((speed_ratio_list[i] + speed_ratio_list[i + 1]) / 2)
    if speed_ratio_cal <= speed_ratio_list_mean[-1]:
        return speed_ratio_list[(len(speed_ratio_list)) - 1]
    gear = 1
    for i in speed_ratio_list_mean:
        if speed_ratio_cal > i:
            return speed_ratio_list[gear - 1]
        gear += 1

def get_gear(speed_ratio_cal):
    speed_ratio_list = gear_list
    speed_ratio_list = sorted(speed_ratio_list, reverse=True)
    speed_ratio_list_mean = []
    for i in range(len(speed_ratio_list) - 1):
        speed_ratio_list_mean.append((speed_ratio_list[i] + speed_ratio_list[i + 1]) / 2)
    if speed_ratio_cal <= speed_ratio_list_mean[-1]:
        return len(speed_ratio_list)
    gear = 1
    for i in speed_ratio_list_mean:
        if speed_ratio_cal > i:
            return gear
        gear += 1

def get_speed_ratio(speed_ratio_cal):

    speed_ratio_list = gear_list
    speed_ratio_list = sorted(speed_ratio_list, reverse=True)
    speed_ratio_list_mean = []
    for i in range(len(speed_ratio_list) - 1):
        speed_ratio_list_mean.append((speed_ratio_list[i] + speed_ratio_list[i + 1]) / 2)
    if speed_ratio_cal <= speed_ratio_list_mean[-1]:
        return float(speed_ratio_list[-1]),len(speed_ratio_list)
    gear = 1
    for i in speed_ratio_list_mean:
        if speed_ratio_cal > i:
            return float(speed_ratio_list[gear - 1]), gear
        gear += 1
        
def first(the_iterable, condition = lambda x: True):#第一份满足条件的元素
    for i in the_iterable:
        if condition(i):
            return i
        
def group_elements_sizes(arry):#计算连续元素个数，并输出第一个大于1的元素个数
    import itertools
    arry_list = arry.to_list()
    items, counts = [], []
    for k, v in itertools.groupby(arry_list):
        items.append(k)
        counts.append(len(list(v)))
    ind_list = [i for i,x in enumerate(items) if x==1]
    elements_size = [counts[j] for j in ind_list]
    
    flag_s = 0
    start_pnt = 0
    if len(elements_size) and max(elements_size)>=2:
        flag_s = first(elements_size, lambda i:i>1)
        elements_id = elements_size.index(flag_s)
        start_id = ind_list[elements_id]
        for n in range(start_id):
            start_pnt+=counts[n] 
    elif len(elements_size) and max(elements_size)==1:
        flag_s = first(elements_size, lambda i:i==1)
        elements_id = elements_size.index(flag_s)
        start_id = ind_list[elements_id]
        for n in range(start_id):
            start_pnt+=counts[n]
    
    return  flag_s,start_pnt

def group_elements_sizes1(arry):#计算连续怠速时长
    import itertools
    arry_list = arry.to_list()
    items, counts = [], []
    for k, v in itertools.groupby(arry_list):
        items.append(k)
        counts.append(len(list(v)))
    ind_list = [i for i,x in enumerate(items) if x==1]
    elements_size = [counts[j] for j in ind_list]
    return  elements_size

#分布柱状图函数
def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'
from matplotlib.ticker import FuncFormatter
def hist_gram(array,step = 10,width = 10,alpha = 1,label = ''):
    counts = np.histogram(array,bins = np.arange(0,(array.max()+step),step))
    x = np.arange(0,array.max(),step)
    y = counts[0]/np.sum(counts[0])
    plt.bar(np.arange(0,array.max(),step),counts[0]/np.sum(counts[0]),width=width,
            alpha = alpha,align = 'edge',label = label)
    stop_time = np.sum(array==0)/len(array)#0值单独统计
    stop_list = [0]*len(counts[0])
    stop_list[0] = stop_time
    #plt.bar(np.arange(0,array.max(),step),stop_list,width=width,align = 'edge') 
    for a,b in zip(x,y):
        plt.text(a+step/2,b,'%.1f%%'%(b*100),ha = 'center',va = 'bottom',fontsize=15)
    plt.xticks(np.arange(0,array.max()+10,step))

    plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.ylabel('占比')

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

#文件导入
#如果有文件上传，显示数据表格
if uploaded_datafile is not None:
    file_name = uploaded_datafile.name
    if file_name.endswith('.xlsx'):
        dfa = pd.read_excel(uploaded_datafile)
    if file_name.endswith('.csv'):
        dfa = pd.read_csv(uploaded_datafile)

    st.write('🟦原始数据：',dfa.head())
    if '.' in dfa.columns[0]:
        dfa.columns = [x.split('.')[1] for x in dfa.columns.tolist()]
    id_list = list(set(dfa.terminalid.values))

    dfa_result = []
    for t_id in id_list:
        df = dfa.loc[dfa.terminalid == t_id].copy()
        df = df.dropna(how = 'all')#删除空行
        #df['gpstime'] = pd.to_datetime(df['gpstime']) #将数据类型转换为日期类型
        df.sort_values(by=['gpstime'],inplace=True) 
        #dfa = dfa.sort_values(by=["gpstime"])  # 按时间排序
        df = df.reset_index(drop=True)  # 重新设置索引
        df['transmissionrotation'] = df.transmissionrotation.rolling(2).median().shift(-1)
        df['transmissionrotation'].fillna(method = 'pad',inplace = True)
        # 滑膜功能开发
        is_0time = 0
        last_is_0time = 0
        # 速比、档位
        df['speed_ratio'] = df['rotation'] / (df['transmissionrotation'] + 1e-5)
        df['speed_ratio_define'] = df['speed_ratio'].apply(lambda x: get_speed_ratio_define(x))
        df['speed_ratio_before'] = df['speed_ratio'].shift(1)
        df.loc[df['speed_ratio_before'].isnull(), 'speed_ratio_before'] = -99

        df['speed_ratio_after'] = df['speed_ratio'].shift(-1)
        df.loc[df['speed_ratio_after'].isnull(), 'speed_ratio_after'] = -99

        df['gear'] = df['speed_ratio'].apply(lambda x: get_gear(x))
        df['gear_after'] = df['gear'].shift(-1)
        df['engineoutputtorque_after'] = df['engineoutputtorque'].shift(-1)
        df['rotation_after'] = df['rotation'].shift(-1)

        df.loc[df['gear_after'].isnull(), 'gear_after'] = 0

        df = df.astype(
            {
                'speed_ratio': 'float', 'speed_ratio_define': 'float', 'speed_ratio_before': 'float', 'speed_ratio_after': 'float',
                'gear': 'int', 'gear_after': 'int'
             })


        # 一轴转速
        df['one_rotation'] = df['transmissionrotation'] * df['speed_ratio_define']
        df['one_rotation_after'] = df['one_rotation'].shift(-1)

        # 滑膜功
        #df['synovial_power'] = df['engineoutputtorque'] * (df['rotation'] - df['one_rotation']) * 1*2240/100/9549

        #global outputs_list
        output_list = []
        for i, row in df.iterrows():
            #result = deepcopy(default)  # 设置初始值
            #result_list = []
            start_end_index = 0
            df_after30 = None
            terminal_id = df["terminalid"][0]

            is_0time = is_0time + 1 if row['instrumentspeed'] == 0 else 0

            if last_is_0time >= 2 and is_0time == 0 and i + 30 < len(df):
                df5 = df.iloc[i: i + 15]
                if len(df5.loc[df5['instrumentspeed'] > 0]) >= 5 and len(df5.loc[df5['accelerator'] > 0]) and len(df5.loc[df5['instrumentspeed'] > 3]):

                    df_after30 = df.iloc[i : i + 30]
                    #df_after30_0 = df.iloc[i+2 : i + 30]

                    df_after30['trans_ratio_0'] = df_after30['transmissionrotation'] * gear_list[0]/df_after30['rotation']
                    df_after30['trans_ratio_diff'] = df_after30['trans_ratio_0'].diff()
                    if list(df_after30['trans_ratio_diff'])[1]>0:
                        last_index1 = df_after30.loc[df_after30['trans_ratio_diff'] <0].index[0]
                    else:
                        last_index1 = df_after30.loc[df_after30['trans_ratio_diff'] <0].index[1]
                    df_after30 = df.iloc[i : last_index1+1].copy()

                    up_gear_list = list(set(df_after30['gear']))
                    flag_s_list = []
                    start_pnt_list = []
                    trans_ratio_diff_list = []
                    trans_ratio_1_diff_list = []
                    trans_ratio_1_diff_list2 = []
                    st_gear_list = []
                    st_gear_list2 = []

                    for gi in up_gear_list:
                        df_after30['trans_ratio'] = df_after30['transmissionrotation'] * gear_list[len(gear_list)-gi]/df_after30['rotation']
                        df_after30['trans_ratio_diff'] = df_after30['trans_ratio'].diff().apply(lambda x: x if x>=0 else 100)
                        df_after30['trans_ratio_flag'] = df_after30['trans_ratio'].apply(lambda x: 1 if (x>0.915)and (x<1.01) else 0)
                        df_after30.at[0,'trans_ratio_flag'] = 0
                        df_after30['trans_ratio_1_diff'] = 1- df_after30['trans_ratio']

                        syn_st_ind = df_after30[df_after30['trans_ratio']<1].index.max()#最接近1的索引
                        syn_st_ind2 = df_after30[df_after30['trans_ratio']>=1].index.min()
                        #syn_st_ind = min(syn_st_ind1,syn_st_ind2)
                        try :
                            trans_ratio_diff_num0 = df_after30['trans_ratio_diff'][syn_st_ind]#上升梯度
                            trans_ratio_diff_num = df_after30['trans_ratio_diff'][syn_st_ind+1]#上升梯度
                            trans_ratio_1_diff_num = df_after30['trans_ratio_1_diff'][syn_st_ind]#与1之间距离
                            flag_s = df_after30['trans_ratio_flag'][syn_st_ind]#近1标记
                        except KeyError:
                            trans_ratio_diff_num0 = 100
                            trans_ratio_diff_num = 100
                            trans_ratio_1_diff_num = 100
                            flag_s = 2

                        if np.logical_not(np.isnan(syn_st_ind)):
                            st_gear_list.append(gi)
                            start_pnt_list.append(syn_st_ind)
                            trans_ratio_diff_list.append(min(trans_ratio_diff_num0,trans_ratio_diff_num))
                            trans_ratio_1_diff_list.append(trans_ratio_1_diff_num)
                            flag_s_list.append(flag_s)

                        elif np.logical_not(np.isnan(syn_st_ind2)) :
                            st_gear_list2.append(gi)
                            trans_ratio_1_diff_list2.append(trans_ratio_1_diff_num)

                    condtion_flag_list = []
                    if len(start_pnt_list)>0:
                        for ix in range(len(start_pnt_list)):
                            condtion_flag = 100
                            a1 = start_pnt_list[ix]
                            b1 = trans_ratio_diff_list[ix]
                            c1 = trans_ratio_1_diff_list[ix]
                            d1 = st_gear_list[ix]
                            e1 = flag_s_list[ix]
                            try :
                                e0 = df_after30['trans_ratio_flag'][syn_st_ind-1]
                                e2 = df_after30['trans_ratio_flag'][syn_st_ind+1]
                            except (IndexError,KeyError):
                                e0 = 0
                                e2 = 0

                            if (e1 ==1) & (e2 == 0):
                                if (c1 == min(trans_ratio_1_diff_list))&(b1 == min(trans_ratio_diff_list)):
                                    condtion_flag = 2
                                elif (c1 == min(trans_ratio_1_diff_list))&(b1 <=0.3)&(b1 >min(trans_ratio_diff_list)):
                                    condtion_flag = 3
                                elif (c1 == min(trans_ratio_1_diff_list))&(b1 >0.3):
                                    condtion_flag = 5   
                            elif (e1 == 0) & (e2 == 0):
                                if (c1 >=0.8)&(b1 == min(trans_ratio_diff_list)):
                                    condtion_flag = 4
                                elif (c1 <0.8)&(b1 == min(trans_ratio_diff_list)):
                                    condtion_flag = 6
                            elif (e1 ==1) & ((e2 == 1)|(e0 == 1)):
                                condtion_flag = 1
                            condtion_flag_list.append(condtion_flag)

                        c_flag_min = condtion_flag_list.index(min(condtion_flag_list))                
                        #st_gear_ind = flag_s_list.index(max(flag_s_list))
                        st_gear = st_gear_list[c_flag_min]
                        start_end_index = start_pnt_list[c_flag_min]    
                    else: 
                        list_id = trans_ratio_1_diff_list2.index(min(trans_ratio_1_diff_list2))
                        st_gear = st_gear_list2[list_id]
                        start_end_index = i

            if start_end_index:
                st_gpstime = df.iloc[i-1:start_end_index + 1]['gpstime'].values[0]
                # result['synovial_power_all'] = float(df.iloc[i: start_end_index + 1]['synovial_power'].sum())
                #st_gear = df.iloc[start_end_index:start_end_index + 1]['gear'].values[0]
                #st_gear = up_gear_list[st_gear_ind]

                # 一轴转速
                df_all = df.iloc[i-1: start_end_index + 1]
                speed_ratio_list = gear_list
                df_all['one_rotation'] = df_all['transmissionrotation'] * speed_ratio_list[int(len(gear_list) - st_gear)]
                # 滑膜功
                #df_all['synovial_power'] = df_all['engineoutputtorque'] * (df_all['rotation'] - df_all['one_rotation'])*2100 / 100 / 9549
                df_all['synovial_power'] = df_all.apply(lambda row: get_synovial_power(row['engineoutputtorque'],
                                                                                       row['engineoutputtorque_after'],
                                                                                       row['rotation'],
                                                                                       row['rotation_after'],
                                                                                       row['one_rotation'],
                                                                                       row['one_rotation_after']),axis = 1)
                st_synovial_power_all = float(df_all.iloc[0:(df_all.index.max()-1)]['synovial_power'].sum())
                # result['synovial_power_all'] = df.iloc[i: start_end_index + 1]
                st_synovial_num = len(df.iloc[i: start_end_index + 1])
                st_start_rotation_mean = float(df.iloc[i: start_end_index + 1]['rotation'].mean())
                st_start_rotation_max = float(df.iloc[i: start_end_index + 1]['rotation'].max())
                #gear_origin = df.iloc[i:start_end_index + 1]['当前档位'].values[0]
                # 记录
                result_list = [terminal_id,st_gpstime,st_synovial_power_all,st_gear,st_synovial_num,
                               st_start_rotation_mean,st_start_rotation_max]
            else :
                result_list = []
            if len(result_list):
                output_list.append(result_list)

            last_is_0time = is_0time
        # 滑膜结束
        # -------
        dfa_result.extend(output_list)

    dfa_result = pd.DataFrame(dfa_result,columns=['id','gpstime','滑膜功','起步档位','同步时间','起步平均转速','起步最大转速'])
    #dfa_result.to_excel(dph+'result.xlsx')

        #添加下载链接
    #mime="application/vnd.ms-excel")
    st.write('🟦计算结果：',dfa_result.head())
    df_xlsx = to_excel(dfa_result)
    st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'df_Result.xlsx')
    
    st.set_option('deprecation.showPyplotGlobalUse', False)#屏蔽警告

    dfa_result['能量区间'] = dfa_result['滑膜功'].apply(lambda x: 'L'if x<100 else('M' if x<200 else 'H') )
    dfa_gear_fenb =  dfa_result['起步档位'].value_counts().sort_index()
    dfa_synovial_power_fenb =  dfa_result['能量区间'].value_counts()

    st.write('🟦可视化结果：',)
    fig1 = plt.figure(figsize=(16,16))
    plt.suptitle("起步行为")

    plt.subplot(221)
    plt.pie(dfa_gear_fenb.values,labels=[str(i) + '档' for i in dfa_gear_fenb.index],autopct='%1.2f%%') 
    plt.title("起步档位");

    plt.subplot(223)
    hist_gram(dfa_result.起步平均转速,step = 100,width = 90)
    plt.xticks(rotation = 45)  
    plt.title('起步平均转速')

    plt.subplot(224)
    hist_gram(dfa_result.起步最大转速,step = 100,width = 90)
    plt.xticks(rotation = 45)
    plt.title('起步最大转速')

    plt.subplot(222)
    plt.pie(dfa_synovial_power_fenb.values,labels=dfa_synovial_power_fenb.index,autopct='%1.2f%%')
    plt.title('滑膜功')

    plt.tight_layout()
    st.pyplot(fig1)


    dfa['open_flag'] = 0
    dfa.loc[(dfa.instrumentspeed>0)&(dfa.rotation>701)&(dfa.clutchswitch==1)&(dfa.accelerator>0),'open_flag'] = 1

    ban_ld = group_elements_sizes1(dfa['open_flag'])
    ban_ld = [n for n in ban_ld if n >= 10 ]

    fig2 = plt.figure(figsize=(16,9))
    hist_gram(np.array(ban_ld),step = 10,width = 9,alpha = 0.6,label = '')
    plt.title('半联动时间分布')
    st.pyplot(fig2)
    