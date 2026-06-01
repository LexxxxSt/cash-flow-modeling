import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# 1. ИНИЦИАЛИЗАЦИЯ
# =========================================================

np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------
# БАЗОВЫЕ ПАРАМЕТРЫ
# ---------------------------------------------------------

NUM_CENTERS = 11
NUM_GROUPS = 345
NUM_TEACHERS = 101

DAYS = [1, 2, 3, 4, 5, 6]  # Пн-Сб
TIME_SLOTS = list(range(1, 8))  # 8 слотов времени

MAX_LESSONS_PER_DAY = 5
MAX_LESSONS_PER_WEEK = 18

REGIONAL_MULTIPLIER = 1.0
NDFL_RATE = 0.13
INSURANCE_RATE = 0.30

LESSON_FIXED_RATE = {
    'school': 600,
    'teen': 570,
    'adult': 550
}

VARIABLE_TEACHER_SHARE = 0.36
WEEKS_IN_MONTH = 4.33  

# ---------------------------------------------------------
# КАТЕГОРИИ
# ---------------------------------------------------------

CATEGORIES = {
    'school': {
        'groups': 191,
        'teachers': 48
    },
    'teen': {
        'groups': 72,
        'teachers': 26
    },
    'adult': {
        'groups': 82,
        'teachers': 27
    }
}

SUBJECTS = ['english', 'chinese', 'other']

# ---------------------------------------------------------
# ЦЕНТРЫ И КАБИНЕТЫ
# ---------------------------------------------------------

center_data = []

for i in range(1, 4):
    center_data.append({
        'center_id': f'Center_{i}',
        'rooms': 15
    })

for i in range(4, 9):
    center_data.append({
        'center_id': f'Center_{i}',
        'rooms': 10
    })

for i in range(9, 12):
    center_data.append({
        'center_id': f'Center_{i}',
        'rooms': 7
    })

centers_df = pd.DataFrame(center_data)
centers_df.to_csv('centers.csv', index=False, encoding='utf-8-sig')

center_rooms = centers_df.set_index('center_id')['rooms'].to_dict()
center_ids = centers_df['center_id'].tolist()

# =========================================================
# 2. ГЕНЕРАЦИЯ ГРУПП (ДИАПАЗОН ПОСЕЩАЕМОСТИ 0.50 - 1.00)
# =========================================================

def teacher_subject_for_category(category):
    if category == 'school':
        return np.random.choice(
            ['english', 'chinese', 'other'],
            p=[0.85, 0.12, 0.03]
        )
    elif category == 'teen':
        return np.random.choice(
            ['english', 'chinese', 'other'],
            p=[0.85, 0.12, 0.03]
        )
    else:  # adult
        return np.random.choice(
            ['english', 'chinese', 'other'],
            p=[0.70, 0.15, 0.15]
        )

def generate_group_size():
    if np.random.rand() < 0.35:
        return np.random.randint(4, 7), 'mini'
    return np.random.randint(7, 16), 'standard'

center_weights = np.array([15,15,15,10,10,10,10,10,7,7,7])
center_weights = center_weights / center_weights.sum()

assigned_centers = np.random.choice(
            center_ids,
            size=NUM_GROUPS,
            p=center_weights
)

groups = []
current_group_id = 1

for category, values in CATEGORIES.items():
    for _ in range(values['groups']):
        subject = teacher_subject_for_category(category)
        group_size, group_type = generate_group_size()

        if group_type == 'mini':
            monthly_price = 5200
        else:
            monthly_price = 4500

        price_per_lesson_student = monthly_price / 8
        center = assigned_centers[current_group_id - 1]

        time_slot = np.random.choice(
            TIME_SLOTS,
            p=[0.10, 0.12, 0.15, 0.18, 0.18, 0.15, 0.12]
        )

        attendance_rate = np.random.uniform(0.50, 1.00)
        if attendance_rate <= 0.50:
            effective_price = monthly_price * 0.50
        else:
            effective_price = monthly_price
        monthly_revenue = group_size * effective_price

        groups.append({
            'group_id': current_group_id,
            'category': category,
            'subject': subject,
            'center': center,
            'group_type': group_type,
            'students': group_size,
            'monthly_price_per_student': monthly_price,
            'price_per_lesson_student': round(price_per_lesson_student, 2),
            'attendance_rate': round(attendance_rate, 2), 
            'monthly_revenue': round(monthly_revenue, 2),
            'time_slot': time_slot
        })
        current_group_id += 1

groups_df = pd.DataFrame(groups)
groups_df.to_csv('groups.csv', index=False, encoding='utf-8-sig')


# =========================================================
# 3. ГЕНЕРАЦИЯ ПРЕПОДАВАТЕЛЕЙ
# =========================================================

teachers = []
teacher_id = 1

for category, values in CATEGORIES.items():
    for _ in range(values['teachers']):
        subject = teacher_subject_for_category(category)
        home_center = np.random.choice(center_ids)
        additional_center = np.random.choice(center_ids)

        teachers.append({
            'teacher_id': teacher_id,
            'category': category,
            'subject': subject,
            'home_center': home_center,
            'additional_center': additional_center,
            'max_work_days': np.random.choice([5,6], p=[0.5,0.5]), 
            'max_lessons_week': np.random.choice([16,18], p=[0.4,0.6])
        })
        teacher_id += 1

teachers_df = pd.DataFrame(teachers)
teachers_df.to_csv('teachers.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 4. ДОСТУПНОСТЬ ПРЕПОДАВАТЕЛЕЙ
# =========================================================

availability = []
for teacher in teachers_df.itertuples():
    for day in DAYS:
        availability.append({
            'teacher_id': teacher.teacher_id,
            'day': day,
            'available': 1
        })
availability_df = pd.DataFrame(availability)
availability_df.to_csv('availability.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 5. PYOMO МОДЕЛЬ
# =========================================================

model = ConcreteModel()

model.G = Set(initialize=groups_df['group_id'].tolist())
model.T = Set(initialize=teachers_df['teacher_id'].tolist())
model.D = Set(initialize=DAYS)
model.S = Set(initialize=TIME_SLOTS)
model.C = Set(initialize=center_ids)

# ПАРАМЕТРЫ
group_category = groups_df.set_index('group_id')['category'].to_dict()
group_subject = groups_df.set_index('group_id')['subject'].to_dict()
group_center = groups_df.set_index('group_id')['center'].to_dict()
group_slot = groups_df.set_index('group_id')['time_slot'].to_dict()

teacher_category = teachers_df.set_index('teacher_id')['category'].to_dict()
teacher_subject = teachers_df.set_index('teacher_id')['subject'].to_dict()
teacher_max_days = teachers_df.set_index('teacher_id')['max_work_days'].to_dict()
teacher_max_week = teachers_df.set_index('teacher_id')['max_lessons_week'].to_dict()

availability_dict = {
    (row.teacher_id, row.day): row.available
    for _, row in availability_df.iterrows()
}

# ПЕРЕМЕННЫЕ
model.x = Var(model.T, model.G, model.D, domain=Binary)
model.z = Var(model.T, model.G, domain=Binary)
model.work_day = Var(model.T, model.D, domain=Binary)

# =========================================================
# 6. ОГРАНИЧЕНИЯ
# =========================================================

def group_lessons_rule(m, g):
    return sum(m.x[t, g, d] for t in m.T for d in m.D) == 2
model.GroupLessons = Constraint(model.G, rule=group_lessons_rule)

def one_teacher_rule(m, g):
    return sum(m.z[t, g] for t in m.T) == 1
model.OneTeacher = Constraint(model.G, rule=one_teacher_rule)

def link_rule(m, t, g, d):
    return m.x[t, g, d] <= m.z[t, g]
model.Link = Constraint(model.T, model.G, model.D, rule=link_rule)

def teacher_group_rule(m, t, g):
    return sum(m.x[t, g, d] for d in m.D) == 2 * m.z[t, g]
model.TeacherGroup = Constraint(model.T, model.G, rule=teacher_group_rule)

def category_rule(m, t, g):
    if teacher_category[t] != group_category[g]:
        return m.z[t, g] == 0
    return Constraint.Skip
model.CategoryMatch = Constraint(model.T, model.G, rule=category_rule)

def subject_rule(m, t, g):
    if teacher_subject[t] != group_subject[g]:
        return m.z[t, g] == 0
    return Constraint.Skip
model.SubjectMatch = Constraint(model.T, model.G, rule=subject_rule)

def availability_rule(m, t, g, d):
    if availability_dict[(t, d)] == 0:
        return m.x[t, g, d] == 0
    return Constraint.Skip
model.Availability = Constraint(model.T, model.G, model.D, rule=availability_rule)

def no_conflict_rule(m, t, d, s):
    groups_in_slot = [g for g in m.G if group_slot[g] == s]
    return sum(m.x[t, g, d] for g in groups_in_slot) <= 1
model.NoConflict = Constraint(model.T, model.D, model.S, rule=no_conflict_rule)

def room_capacity_rule(m, c, d, s):
    groups_here = [g for g in m.G if group_center[g] == c and group_slot[g] == s]
    if len(groups_here) == 0:
        return Constraint.Skip
    return sum(m.x[t, g, d] for t in m.T for g in groups_here) <= center_rooms[c]
model.RoomCapacity = Constraint(model.C, model.D, model.S, rule=room_capacity_rule)

def max_daily_rule(m, t, d):
    return sum(m.x[t, g, d] for g in m.G) <= MAX_LESSONS_PER_DAY
model.MaxDaily = Constraint(model.T, model.D, rule=max_daily_rule)

def max_week_rule(m, t):
    return sum(m.x[t, g, d] for g in m.G for d in m.D) <= teacher_max_week[t]
model.MaxWeek = Constraint(model.T, rule=max_week_rule)

def work_day_rule(m, t, d):
    return sum(m.x[t, g, d] for g in m.G) <= m.work_day[t, d] * MAX_LESSONS_PER_DAY
model.WorkDay = Constraint(model.T, model.D, rule=work_day_rule)

def max_days_rule(m, t):
    return sum(m.work_day[t, d] for d in m.D) <= teacher_max_days[t]
model.MaxDays = Constraint(model.T, rule=max_days_rule)

# =========================================================
# 7. ЦЕЛЕВАЯ ФУНКЦИЯ
# =========================================================

model.Objective = Objective(
    expr=sum(model.work_day[t, d] for t in model.T for d in model.D) +
         0.01 * sum(model.x[t, g, d] for t in model.T for g in model.G for d in model.D),
    sense=minimize
)

# =========================================================
# 8. РЕШЕНИЕ
# =========================================================

solver = SolverFactory('appsi_highs')
solver.options['time_limit'] = 60          
solver.options['mip_rel_gap'] = 0.08  

print("Запуск решателя...")
result = solver.solve(model, tee=False, load_solutions=False)

if result.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.feasible, TerminationCondition.maxTimeLimit]:
    model.solutions.load_from(result)
    print("Решение успешно загружено в модель.")
else:
    raise RuntimeError("Решателю не удалось найти допустимое расписание.")

# =========================================================
# 9. СОХРАНЕНИЕ РАСПИСАНИЯ
# =========================================================

schedule = []
for t in model.T:
    for g in model.G:
        if model.z[t, g].value is not None and model.z[t, g].value > 0.5:
            for d in model.D:
                if model.x[t, g, d].value is not None and model.x[t, g, d].value > 0.5:
                    row = groups_df[groups_df['group_id'] == g].iloc[0]
                    schedule.append({
                        'teacher_id': t, 'group_id': g, 'day': d, 'slot': row['time_slot'], 'center': row['center'],
                        'category': row['category'], 'subject': row['subject'], 'students': row['students'],
                        'price_per_lesson_student': row['price_per_lesson_student'], 'attendance_rate': row['attendance_rate'],
                        'monthly_revenue': row['monthly_revenue']
                    })

schedule_df = pd.DataFrame(schedule)
schedule_df.to_csv('schedule_detailed.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 10. РАСЧЕТ ФОТ (ПОСТУДЕНЧЕСКИЙ KPI)
# =========================================================

teacher_payments = []
for teacher_id in teachers_df['teacher_id']:
    teacher_schedule = schedule_df[schedule_df['teacher_id'] == teacher_id]
    if len(teacher_schedule) == 0:
        continue

    category = teachers_df[teachers_df['teacher_id'] == teacher_id]['category'].iloc[0]
    lessons_per_month = len(teacher_schedule) * WEEKS_IN_MONTH
    fixed_payment = lessons_per_month * LESSON_FIXED_RATE[category]

    variable_payment = 0
    unique_groups = teacher_schedule['group_id'].unique()

    for g_id in unique_groups:
        g_data = groups_df[groups_df['group_id'] == g_id].iloc[0]
        lessons_week_for_group = len(teacher_schedule[teacher_schedule['group_id'] == g_id])
        lessons_month_for_group = lessons_week_for_group * WEEKS_IN_MONTH
        
        group_bonus = (
            VARIABLE_TEACHER_SHARE 
            * g_data['price_per_lesson_student'] 
            * g_data['students'] 
            * g_data['attendance_rate'] 
            * lessons_month_for_group
        )
        variable_payment += group_bonus

    gross_payment = fixed_payment + variable_payment
    regional_payment = gross_payment * REGIONAL_MULTIPLIER
    ndfl = regional_payment * NDFL_RATE
    insurance = regional_payment * INSURANCE_RATE

    teacher_payments.append({
        'teacher_id': teacher_id, 'category': category, 'lessons_per_week': len(teacher_schedule), 'groups_count': len(unique_groups),
        'fixed_payment': round(fixed_payment, 2), 'variable_payment': round(variable_payment, 2),
        'gross_payment': round(gross_payment, 2), 'regional_payment': round(regional_payment, 2),
        'ndfl': round(ndfl, 2), 'insurance': round(insurance, 2), 'net_company_cost': round(regional_payment + insurance, 2)
    })

teacher_payments_df = pd.DataFrame(teacher_payments)
teacher_payments_df.to_csv('teacher_payments.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 11. СТАТИСТИКА ЦЕНТРОВ
# =========================================================

center_stats = groups_df.groupby('center').agg(
    total_groups=('group_id', 'count'),
    total_students=('students', 'sum'),
    total_revenue=('monthly_revenue', 'sum')
).reset_index()

center_stats['rooms'] = center_stats['center'].map(center_rooms)
center_stats['room_utilization'] = center_stats['total_groups'] / (center_stats['rooms'] * len(TIME_SLOTS) * len(DAYS))
center_stats.to_csv('center_statistics.csv', index=False, encoding='utf-8-sig')

groups_by_center_cat = groups_df.groupby(['center', 'category']).size().unstack(fill_value=0)
groups_by_center_cat['total_groups'] = groups_by_center_cat.sum(axis=1)

teacher_center = schedule_df[['teacher_id', 'center']].drop_duplicates()
unique_teachers_per_center = teacher_center.groupby('center')['teacher_id'].nunique()
unique_teachers_per_center.name = 'unique_teachers'

teacher_monthly_cost = teacher_payments_df['net_company_cost'].copy()
teacher_monthly_cost.index = teacher_payments_df['teacher_id']

lessons_teacher_center = schedule_df.groupby(['teacher_id', 'center']).size().reset_index(name='lessons')
total_lessons_teacher = lessons_teacher_center.groupby('teacher_id')['lessons'].sum()
lessons_teacher_center = lessons_teacher_center.merge(total_lessons_teacher.rename('total_lessons'), on='teacher_id')
lessons_teacher_center['share'] = lessons_teacher_center['lessons'] / lessons_teacher_center['total_lessons']
lessons_teacher_center['monthly_cost'] = lessons_teacher_center['teacher_id'].map(teacher_monthly_cost)
lessons_teacher_center['cost_in_center'] = lessons_teacher_center['monthly_cost'] * lessons_teacher_center['share']

center_fot_month = lessons_teacher_center.groupby('center')['cost_in_center'].sum()
center_fot_week = (center_fot_month / 4).round(1)
center_fot_week.name = 'weekly_payroll'

teacher_load = teacher_payments_df[['teacher_id', 'lessons_per_week']].merge(teachers_df[['teacher_id', 'max_lessons_week']], on='teacher_id')
teacher_load['load_pct'] = (teacher_load['lessons_per_week'] / teacher_load['max_lessons_week']) * 100

teacher_center_load = teacher_center.merge(teacher_load[['teacher_id', 'load_pct']], on='teacher_id')
avg_load_per_center = teacher_center_load.groupby('center')['load_pct'].mean().round(1)
avg_load_per_center.name = 'avg_teacher_load'

center_stats_detailed = groups_by_center_cat.copy().join(unique_teachers_per_center).join(center_fot_week).join(avg_load_per_center)
center_stats_detailed = center_stats_detailed.rename(columns={
    'school': 'school_groups', 'teen': 'teen_groups', 'adult': 'adult_groups',
    'unique_teachers': 'teachers_count', 'weekly_payroll': 'weekly_payroll', 'avg_teacher_load': 'avg_load_pct'
}).reset_index()

summary_row = {
    'center': 'Итого', 'school_groups': center_stats_detailed['school_groups'].sum(), 'teen_groups': center_stats_detailed['teen_groups'].sum(),
    'adult_groups': center_stats_detailed['adult_groups'].sum(), 'total_groups': center_stats_detailed['total_groups'].sum(),
    'teachers_count': teachers_df['teacher_id'].nunique(), 'weekly_payroll': round(teacher_payments_df['net_company_cost'].sum() / 4, 1),
    'avg_load_pct': round((teacher_load['lessons_per_week'].sum() / teacher_load['max_lessons_week'].sum()) * 100, 1)
}
center_stats_detailed = pd.concat([center_stats_detailed, pd.DataFrame([summary_row])], ignore_index=True)
center_stats_detailed.to_csv('center_stats_detailed.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 12. ДЕТЕРМИНИРОВАННЫЙ CASH FLOW (БАЗОВЫЙ СЦЕНАРИЙ)
# =========================================================

seasonality = {1: 0.95, 2: 1.00, 3: 1.05, 4: 1.05, 5: 1.00, 6: 0.75, 7: 0.30, 8: 0.20, 9: 1.25, 10: 1.10, 11: 1.05, 12: 1.00}

monthly_base_revenue = groups_df['monthly_revenue'].sum()
monthly_base_fixed_payroll = teacher_payments_df['fixed_payment'].sum()
monthly_base_variable_payroll = teacher_payments_df['variable_payment'].sum()

cash_flows = []
for month in range(1, 13):
    season_factor = seasonality[month]
    
    revenue = monthly_base_revenue * season_factor
    fixed_payroll = monthly_base_fixed_payroll * season_factor
    variable_payroll = monthly_base_variable_payroll * season_factor
    
    total_payroll = fixed_payroll + variable_payroll
    insurance = total_payroll * INSURANCE_RATE
    net_flow = revenue - total_payroll - insurance # НДФЛ не вычитается, он уже внутри payroll

    cash_flows.append({
        'month': month, 'revenue': round(revenue, 2), 'payroll': round(total_payroll, 2),
        'insurance': round(insurance, 2), 'net_cash_flow': round(net_flow, 2)
    })

cash_flow_df = pd.DataFrame(cash_flows)
cash_flow_df.to_csv('cash_flows.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 13. ИМИТАЦИОННОЕ МОДЕЛИРОВАНИЕ МОНТЕ-КАРЛО (ИСПРАВЛЕНО)
# =========================================================

scenarios = []
STARTING_CASH_RESERVE = 1200000 
FIXED_OVERHEAD_MONTHLY = 4620000.0  # Постоянные затраты сети (аренда 11 центров, адм. расходы)
NUM_SIMULATIONS = 1000

# Массивы для построения графиков
all_trajectories = np.zeros((NUM_SIMULATIONS, 13))
all_trajectories[:, 0] = STARTING_CASH_RESERVE
monthly_gap_counts = {m: 0 for m in range(1, 13)}

total_gaps = 0

for s in range(NUM_SIMULATIONS):
    cumulative_cash = STARTING_CASH_RESERVE
    min_cash = cumulative_cash
    cash_gap_triggered = 0

    for month in range(1, 13):
        season_factor = seasonality[month]
        
        # 1. Случайный коммерческий шок спроса (Этап 2 диплома)
        xi_rev = np.random.normal(1.0, 0.12)
        # 2. Случайный операционный шок издержек (Этап 2 диплома)
        xi_cost = np.random.normal(1.0, 0.05)
        
        # 3. Случайный шок фактической посещаемости
        attendance_shock = np.random.uniform(0.50, 1.00)
        
        # ЮРИДИЧЕСКИЙ ТРИГГЕР ПЕРЕРАСЧЕТА (Исправлено условие диапазона)
        if attendance_shock <= 0.50:
            revenue = (monthly_base_revenue * season_factor) * xi_rev * 0.50
        else:
            revenue = (monthly_base_revenue * season_factor) * xi_rev
        
        simulated_fixed_payroll = (monthly_base_fixed_payroll * season_factor) * xi_cost
        # Переменная часть ФОТ зависит от коэффициента посещаемости
        simulated_variable_payroll = (monthly_base_variable_payroll * season_factor) * (attendance_shock / 0.75) * xi_cost
        
        total_payroll = simulated_fixed_payroll + simulated_variable_payroll
        insurance = total_payroll * INSURANCE_RATE
        
        # Полные стохастические издержки на персонал + постоянное бремя аренды сети
        total_costs = (total_payroll + insurance) + FIXED_OVERHEAD_MONTHLY

        # РАЗНОСТНОЕ УРАВНЕНИЕ ТРАНСФОРМАЦИИ КАПИТАЛА (Этап 3 диплома)
        cumulative_cash += revenue - total_costs
        all_trajectories[s, month] = cumulative_cash
        
        if cumulative_cash < min_cash:
            min_cash = cumulative_cash
            
        if cumulative_cash < 0:
            if cash_gap_triggered == 0:
                monthly_gap_counts[month] += 1
                cash_gap_triggered = 1

    if cash_gap_triggered == 1:
        total_gaps += 1

    scenarios.append({
        'scenario': s, 'final_cash': round(cumulative_cash, 2),
        'min_cash': round(min_cash, 2), 'cash_gap': cash_gap_triggered
    })

monte_carlo_df = pd.DataFrame(scenarios)
monte_carlo_df.to_csv('monte_carlo_results.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 14. RISK REPORT & SUMMARY
# =========================================================

gap_probability = (total_gaps / NUM_SIMULATIONS) * 100

risk_report = pd.DataFrame({
    'metric': ['cash_gap_probability', 'average_final_cash', 'worst_case_cash'],
    'value': [round(gap_probability, 2), round(monte_carlo_df['final_cash'].mean(), 2), round(monte_carlo_df['min_cash'].min(), 2)]
})
risk_report.to_csv('risk_report.csv', index=False, encoding='utf-8-sig')

summary = {
    'groups_total': len(groups_df), 'teachers_total': len(teachers_df), 'students_total': groups_df['students'].sum(),
    'monthly_revenue_total': round(groups_df['monthly_revenue'].sum(), 2),
    'monthly_payroll_total': round(teacher_payments_df['regional_payment'].sum(), 2),
    'cash_gap_probability_percent': round(gap_probability, 2)
}
pd.DataFrame([summary]).to_csv('summary_report.csv', index=False, encoding='utf-8-sig')

# =========================================================
# 15. ВИЗУАЛИЗАЦИЯ ДЛЯ ГЛАВЫ 3 (РИСК-АНАЛИЗ МОНТЕ-КАРЛО)
# =========================================================
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# График 1: Веерное поле 1000 траекторий кумулятивного капитала
plt.figure(figsize=(11, 6))
months_axis = np.arange(13)
for s in range(min(NUM_SIMULATIONS, 200)): # отобразим 200 линий для читаемости веера
    plt.plot(months_axis, all_trajectories[s, :] / 1e6, color='skyblue', alpha=0.3, lw=0.8)
plt.plot(months_axis, np.mean(all_trajectories, axis=0) / 1e6, color='darkblue', lw=2.5, label='Математическое ожидание')
plt.axhline(0, color='red', linestyle='--', lw=1.5, label='Линия банкротства (Cash = 0)')
plt.title('Траектории кумулятивного капитала (1000 симуляций Монте-Карло)', fontsize=12, fontweight='bold')
plt.xlabel('Месяц планирования', fontsize=10)
plt.ylabel('Остаток капитала на счете, млн руб.', fontsize=10)
plt.xlim(0, 12)
plt.xticks(months_axis)
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('monte_carlo_trajectories.png')
plt.close()

# График 2: Вероятность кассового разрыва по календарным месяцам года
plt.figure(figsize=(11, 5))
gap_pct_by_month = [(monthly_gap_counts[m] / NUM_SIMULATIONS) * 100 for m in range(1, 13)]
sns.barplot(x=list(range(1, 13)), y=gap_pct_by_month, palette='Reds_r')
plt.title('Распределение вероятности возникновения кассового разрыва по месяцам года', fontsize=12, fontweight='bold')
plt.xlabel('Номер календарного месяца', fontsize=10)
plt.ylabel('Вероятность локального сбоя ликвидности, %', fontsize=10)
plt.ylim(0, max(gap_pct_by_month) + 2 if max(gap_pct_by_month) > 0 else 5)
plt.tight_layout()
plt.savefig('monthly_gap_probability.png')
plt.close()

# График 3: Распределение итогового кумулятивного капитала (Гистограмма + плотность KDE)
plt.figure(figsize=(11, 5))
sns.histplot(monte_carlo_df['final_cash'] / 1e6, kde=True, color='teal', bins=30, stat='density')
plt.title('Плотность распределения вероятностей итогового годового капитала сети', fontsize=12, fontweight='bold')
plt.xlabel('Финальный капитал компании на конец 12-го месяца, млн руб.', fontsize=10)
plt.ylabel('Плотность частоты распределения', fontsize=10)
plt.tight_layout()
plt.savefig('final_cash_distribution_kde.png')
plt.close()

# График 4: Стек-бар диаграмма структуры затрат на персонал по месяцам
plt.figure(figsize=(11, 5))
months_list = list(range(1, 13))
fixed_costs_arr = np.array([monthly_base_fixed_payroll * seasonality[m] for m in months_list])
var_costs_arr = np.array([monthly_base_variable_payroll * seasonality[m] for m in months_list])
ins_costs_arr = (fixed_costs_arr + var_costs_arr) * INSURANCE_RATE

plt.bar(months_list, fixed_costs_arr / 1e6, label='Фиксированная часть ФОТ преподавателей', color='#4f81bd')
plt.bar(months_list, var_costs_arr / 1e6, bottom=fixed_costs_arr / 1e6, label='Стимулирующий постуденческий KPI (36%)', color='#c0504d')
plt.bar(months_list, ins_costs_arr / 1e6, bottom=(fixed_costs_arr + var_costs_arr) / 1e6, label='Страховые начисления (30%)', color='#9bbb59')

plt.title('Структура затрат сети на обеспечение персонала по месяцам года', fontsize=12, fontweight='bold')
plt.xlabel('Номер календарного месяца', fontsize=10)
plt.ylabel('Сумма операционных выплат, млн руб.', fontsize=10)
plt.xticks(months_list)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('personnel_costs_structure.png')
plt.close()

print(f'\n[УСПЕХ] Скрипт полностью выполнен. Ошибки Монте-Карло устранены.')
print(f'Интегральная вероятность кассового разрыва (P_gap): {round(gap_probability, 2)}%')
for m in range(1, 13):
    print(f' -> Месяц {m}: {round((monthly_gap_counts[m]/NUM_SIMULATIONS)*100, 2)}%')