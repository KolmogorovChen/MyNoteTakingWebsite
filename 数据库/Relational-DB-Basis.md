## 一、 Relational Model and Relational Algebra

**关系模型（Relational Model）：数据怎样组织、应满足什么规则**

**关系代数（Relational Algebra）：怎样对这些数据进行查询和组合**

两者构成关系型数据库的基础。SQL 中的表、主键、筛选、连接等概念，都可以通过它们来理解；后续的查询处理与查询优化，也会用到关系代数。

下面始终使用同一个例子：学校的学生、课程和选课数据。

### 1. 关系模型

先给出三张表。

学生关系 `Student`：

| sid  | name | major            |
| ---- | ---- | ---------------- |
| S1   | 张三 | Statistics       |
| S2   | 李四 | Computer Science |
| S3   | 王五 | Statistics       |
| S4   | 赵六 | Mathematics      |

课程关系 `Course`：

| cid  | title            | credits |
| ---- | ---------------- | ------- |
| C1   | Database         | 3       |
| C2   | Machine Learning | 4       |

选课关系 `Enrollment`：

| sid  | cid  | grade |
| ---- | ---- | ----- |
| S1   | C1   | 90    |
| S1   | C2   | 85    |
| S2   | C1   | 88    |
| S3   | C2   | 92    |

这里，`Enrollment` 中的 `(S1, C1, 90)` 表示：学生 S1 选修课程 C1，成绩为 90。

#### 1.1 **关系的数学定义**

先设有 $n$ 个取值域：

$$
D_1,D_2,\ldots,D_n.
$$

它们的笛卡尔积为：

$$
D_1\times D_2\times\cdots\times D_n = \{(a_1,\ldots,a_n):a_i\in D_i\}.
$$

一个 $n$ 元关系 $r$ 是这个笛卡尔积的一个子集：

$$
\boxed{r\subseteq D_1\times D_2\times\cdots\times D_n.}
$$

对于学生关系：

$$
Student \subseteq D_{\text{sid}} \times D_{\text{name}} \times D_{\text{major}}.
$$
例如：

$$
(S1,\text{张三},\text{Statistics}) \in Student.
$$

笛卡尔积描述所有允许组合的范围；关系只包含数据库当前记录的那些组合。

---------------------

#### 1.2 **基本术语**

| 术语             | 含义                           | 例子                            |
| ---------------- | ------------------------------ | ------------------------------- |
| 关系 Relation    | 元组组成的集合，通常表示成表   | `Student`                       |
| 属性 Attribute   | 一个有名称的字段，通常表示成列 | `major`                         |
| 元组 Tuple       | 关系中的一条记录，通常表示成行 | `(S1, 张三, Statistics)`        |
| 域 Domain        | 某个属性允许取值的集合         | 成绩域可以是 $[0,100]$ 内的数值 |
| 度 Degree／Arity | 属性的数量                     | `Student` 的度为 3              |
| 基数 Cardinality | 元组的数量                     | 当前 `Student` 的基数为 4       |

-------------------

#### 1.3 **关系模式与关系实例**

这两个概念需要明确区分。

**关系模式（Relation Schema）**描述关系的结构，例如：

$$
\text{Student(sid,name,major)}.
$$

完整的模式信息还可以包括属性的域及相关约束。

**关系实例（Relation Instance）**描述某个时刻，关系中实际包含的元组。

例如：

$$
r_{\text{Student}} = \{ (S1,\text{张三},\text{Statistics}), (S2,\text{李四},\text{Computer Science}), \ldots \}.
$$

新增一名学生会改变关系实例，但通常不会改变关系模式；增加一个 `birthday` 属性则会改变模式。

> 很多教材使用大写 **$R$** 表示**关系模式**、小写 **$r$** 表示**关系实例**。但为了简便，也常直接用 `Student` 同时指代关系及其当前内容，需要根据上下文判断。

-------------------------

#### 1.4 经典关系模型中的几个重要性质

- **关系是集合，因此不包含完全重复的元组**

- **元组之间没有固有顺序**

  数据库中的“第一行”“第二行”不是关系本身的逻辑属性。

  如果需要按照成绩从高到低展示，必须额外指定排序。后面会看到，SQL 使用 `ORDER BY` 表达这一要求。

- **属性通常按名称识别**

  写成 $\text{(sid,name,major)}$ 是一种表示约定；在以属性名定义的关系模型中，交换展示列的顺序，不会改变这些属性所描述的事实。进行集合运算时，仍需明确属性如何对应。

- **在经典第一范式的关系模型中，每个属性位置保存一个来自其域的单值**

  “单值”取决于模型对域的定义，可以理解为：不要把需要独立管理的一组事实塞进同一个字段。

---------------------

### 2. 键与约束

只有表结构还不够。数据库还需要回答：

> 哪些属性能够唯一识别一条记录？不同关系之间的引用是否合法？

#### 2.1  超键、候选键与主键

设关系模式 $R$ 的属性集合为 $U$，取属性子集 $K\subseteq U$。

如果在所有合法实例中，任意两个元组只要在 $K$ 上相同，就必须是同一个元组，即：

$$
\forall t_1,t_2\in r,\qquad t_1[K]=t_2[K]\Rightarrow t_1=t_2,
$$

那么 $K$ 是一个**超键（Superkey）**。这里 $t[K]$ 表示元组 $t$ 在属性集合 $K$ 上的取值。

假设学号唯一，那么 $\{\text{sid}\}$ 是超键；同时 $\{\text{sid,name}\}$ 也是超键，因为学号本身已经足以识别学生。

如果一个超键不能再删去任何属性而仍保持唯一性，它就是**候选键（Candidate Key）**。

因此：

- $\{\text{sid}\}$ 是候选键。
- $\{\text{sid,name}\}$ 不是候选键，因为 `name` 是多余的。

这里的“最小”指**没有冗余属性**，不要求所有候选键具有相同数量的属性。

从候选键中选定一个作为主要标识，就是**主键（Primary Key）**。

> 键是模式层面的约束，不能仅凭当前几行数据“碰巧没有重复”就确定。

例如，当前学生姓名可能没有重复，但这不意味着姓名适合作为候选键。

------------------

#### 2.2 复合键

假设每名学生对每门课程最多有一条选课记录，那么 `Enrollment` 可以使用：

$$
\{\text{sid,cid}\}
$$

作为复合主键。原因是：

- `sid` 单独不唯一：一个学生可以选多门课。
- `cid` 单独不唯一：一门课可以有多个学生。
- 两者组合能够唯一确定一条选课记录。

如果允许同一学生在不同学期重修，则需要引入学期或具体开课编号，再重新确定键。

---------------------------------

#### 2.3 **外键与参照完整性**

`Enrollment.sid` 引用 `Student.sid`，表示每条选课记录必须对应一个存在的学生。

在不考虑空值的情况下，可以写成：

$$
\pi_{sid}(Enrollment) \subseteq \pi_{sid}(Student),
$$

其中 $\pi_{sid}$ 表示“取出 `sid` 属性”，后面会正式介绍。

同样：

$$
\pi_{cid}(Enrollment) \subseteq \pi_{cid}(Course).
$$

这就是**参照完整性（Referential Integrity）**的基本思想。

外键本身不要求唯一：多个选课记录可以引用同一个学生。

----------------------------

#### 2.4 **常见完整性约束**

| 约束       | 表达的要求                         |
| ---------- | ---------------------------------- |
| 域约束     | 属性值必须属于允许的域             |
| 键约束     | 键值不能重复                       |
| 实体完整性 | 主键各属性不能为 `NULL`            |
| 参照完整性 | 外键引用必须符合所引用关系的键约束 |
| 业务约束   | 例如成绩范围、课程容量等           |

因此，一个合法数据库状态 $D$ 必须满足预先定义的约束集合 $\mathcal C$：

$$
D\models\mathcal C.
$$

你可以把模式理解为“允许哪些结构”，把约束理解为“允许哪些状态”。

------------------------------------

### 3. 关系代数

**核心概念：用运算表达查询**

关系代数是一套以关系为输入、以关系为输出的运算体系。

例如：

$$
R\xrightarrow{\text{筛选}}R_1 \xrightarrow{\text{取列}}R_2.
$$

**每一步的结果仍然是关系，因此可以继续作为下一步的输入。**这称为关系代数的封闭性，也是复杂查询能够逐步组合的基础。

下面介绍经典关系代数中常用的基本运算：

| 运算                       | 符号     | 作用                                   |
| -------------------------- | -------- | -------------------------------------- |
| 选择 Selection             | $\sigma$ | 按条件筛选元组                         |
| 投影 Projection            | $\pi$    | 保留指定属性                           |
| 并 Union                   | $\cup$   | 合并两个关系的元组                     |
| 差 Difference              | $-$      | 保留属于第一个但不属于第二个关系的元组 |
| 笛卡尔积 Cartesian Product | $\times$ | 将两个关系中的元组两两组合             |
| 重命名 Rename              | $\rho$   | 修改关系名或属性名                     |

连接、交、除法等运算是**由基本运算组合导出的"扩展运算**，但因为非常常用，通常单独讨论。

--------------

#### 3.1 **选择：按条件筛选行**

$\sigma_{\theta}(R)$ 表示在“关系 $R$ 中选择满足条件 $\theta$ 的元组，**把它们收集起来构成一个新的关系实例**”，其数学定义为：

$$
\sigma_{\theta}(R) = \{t\in R:\theta(t)\text{ 为真}\}.
$$

例如，查询统计学专业的学生：
$$
\sigma_{major='Statistics'}(Student).
$$

结果：

| sid  | name | major      |
| ---- | ---- | ---------- |
| S1   | 张三 | Statistics |
| S3   | 王五 | Statistics |

选择运算保留原来的属性结构，只改变元组集合：

$$
|\sigma_{\theta}(R)|\le |R|.
$$

-----------------------------------

#### 3.2 投影：保留指定列

$$
\pi_{A_1,\ldots,A_k}(R) = \{ t[A_1, A_2, ..., A_k] | t \in R \}
$$

- $R$ : 输入关系
- $A_1, \cdots,A_k$：$R$ 的属性，是要保留的字段
- $t \in R$：$t$ 是 $R$ 中的一个元组
- $t[A_1,\dots,A_k]$：从元组 t 里只取出这几个属性的值，构成一个新的更小的元组

> 注意：**投影会自动去重** 
>
> 如果两个不同原元组，截取 $A_1,\dots,A_k$ 之后的值完全一样，在结果集合里只保留一份。 （关系是集合，集合元素不能重复）

因此：

$$
|\pi_A(R)|\le |R|.
$$

选择与投影可以组合：

$$
\boxed{ \pi_{name} \left( \sigma_{major='Statistics'}(Student) \right) }
$$

含义是：

1. 先筛选统计学专业学生。
2. 再取出姓名。

对应 SQL 为：

```sql
SELECT DISTINCT name
FROM Student
WHERE major = 'Statistics';
```

这里加 `DISTINCT` 是为了对应经典关系代数的集合语义。

特别注意术语差异：

- 关系代数的 **Selection** 对应 SQL 的 `WHERE` 筛选。
- 关系代数的 **Projection** 对应 SQL 中指定输出列的部分。

--------------------

#### 3.3 **并、交、差：对元组集合做运算**

设：

$$
A=\pi_{sid}\left(\sigma_{cid='C1'}(Enrollment)\right) =\{S1,S2\}, \\[10pt] B=\pi_{sid}\left(\sigma_{cid='C2'}(Enrollment)\right) =\{S1,S3\}.
$$

为了简洁，这里用学号集合表示单列关系。

那么：

| 表达式    | 结果           | 含义                     |
| --------- | -------------- | ------------------------ |
| $A\cup B$ | $\{S1,S2,S3\}$ | 选了 C1 或 C2 的学生     |
| $A\cap B$ | $\{S1\}$       | 两门课都选了的学生       |
| $A-B$     | $\{S2\}$       | 选了 C1 但没选 C2 的学生 |
| $B-A$     | $\{S3\}$       | 选了 C2 但没选 C1 的学生 |

这些运算要求两个关系**并相容（Union-compatible）**：属性数量相同，对应属性具有兼容的域；在按属性名组织的表示法中，还需对齐属性名。

另外，交运算可以通过差运算得到：

$$
R\cap S=R-(R-S).
$$

差运算尤其适合表达“没有”：

$$
\pi_{sid}(Student)-\pi_{sid}(Enrollment)
$$

表示没有任何选课记录的学生学号，结果为 $\{S4\}$.

--------------

#### 3.4 **笛卡尔积：两两组合**

将 $R$ 的每个元组与 $S$ 的每个元组组合：

$$
R\times S = \{(r,s):r\in R,\ s\in S\}.
$$

若 $|R|=m, |S|=n$，则 $|R\times S|=mn$

例如 $Student\times Course$，包含 $4\times3=12$ 个元组，即所有学生与所有课程的组合。

如果两个关系有同名属性，元组是左右直接拼接，同名属性**全部保留两份**。

---------------------

#### 3.5 重命名：明确属性的身份

重命名用于修改关系名或属性名，常写作：

$$
\rho_{S}(Student)
$$

表示将关系临时命名为 $S$。

也可以将投影结果中的 `sid` 改为 `student_id`：

$$
\rho_{T(student\_id)} \left(\pi_{sid}(Student)\right).
$$

重命名在一个关系需要使用两次时尤其重要。

例如，查找“**专业相同的不同学生对**”，需要把 `Student` 看成两个副本：

$$
S_1=\rho_{S_1}(Student), \qquad S_2=\rho_{S_2}(Student).
$$

然后对 $S_1\times S_2$ 施加条件：

$$
S_1.major=S_2.major \land S_1.sid\ne S_2.sid.
$$

这就是自连接的基本思路。若还要消除 $(S1,S3)$ 与 $(S3,S1)$ 这种对称重复，可以在学号具有可比较顺序时使用 $S_1.sid<S_2.sid$。

----------------------

### 4. 连接

关系模型通常将不同类型的事实分开存储：

- `Student` 存学生信息。
- `Course` 存课程信息。
- `Enrollment` 存选课与成绩。

查询时，经常需要重新组合这些信息。**连接（Join）就是根据相关条件匹配并组合元组。**

---------

#### 4.1 $\theta$ 连接

$\bowtie_\theta$ 读作 theta-join，**条件连接**，定义为：

$$
\boxed{ R\bowtie_{\theta}S = \sigma_{\theta}(R\times S) }. 
$$

逻辑上可以理解为：

1. 先算笛卡尔积：$R\times S$ 

   把 R 的每一行和 S 的每一行**全部拼接（按列合并）**，生成所有组合，不管属性是否重复。

2. 再做选择运算 $\boldsymbol{\sigma_\theta(\dots)}$： 在笛卡尔积结果里，**只保留满足条件 $\theta$ 的那些拼接后的元组**。

> **θ 连接 = 笛卡尔积之后，筛选符合条件的行**。

例如：
$$
Student \bowtie_{Student.sid=Enrollment.sid} Enrollment.
$$

表示将表 $Student$ 与表 $Enrollment$ 全部按列拼接起来，然后只保留其中学号相同的元组。

> 注意：结果里**两列 sid 都会保留**：Student.sid 和 Enrollment.sid，两个同名字段都存在。

对应的 SQL：

```sql
SELECT *
FROM Student, Enrollment
WHERE Student.sid = Enrollment.sid
```

虽然定义使用笛卡尔积，**实际执行时 DBMS 不必先生成完整笛卡尔积**。它可以用哈希连接、索引连接等算法直接寻找匹配项。

-------------------------

#### 4.2 等值连接与自然连接

**等值连接（Equijoin）**是 $\boldsymbol{\theta}$ 连接的特例：连接条件 $\theta$ 是相等比较（`=`）
$$
R \bowtie_{A=B} S = \sigma_{A=B}(R\times S)
$$

其中：A 是 R 的属性，B 是 S 的属性。

**自然连接（Natural Join）**是**特殊的等值连接**，自动用全部同名属性做等值匹配，并且**自动删除重复的同名属性**：

1. 找出 $R$ 和 $S$ 中所有名称相同的属性；
2. 做等值连接，匹配条件：同名属性全部相等；
3. 把重复的同名字段只保留一份。

公式表达：

$$
R\bowtie S = \pi_{\text{全部属性，去重}} \big(\sigma_{R.A_1=S.A_1 \land R.A_2=S.A_2\cdots}(R\times S)\big)
$$

沿用刚才的例子，`Student` 与 `Enrollment` 只有 `sid` 同名，因此：

$$
Student\bowtie Enrollment
$$

会根据 `sid` 匹配，**去掉重复的一列 `sid`**，只保留一个 `sid`。

对应 SQL：

```sql
SELECT Student.sid, name, cid
FROM Student, Enrollment
WHERE Student.sid = Enrollment.sid;
```

自然连接依赖属性命名。如果两个表后来又增加了同名但含义不同的属性，自然连接的含义可能随之变化。因此，理解自然连接时，必须先检查两个关系的全部同名属性。

-----------------------------

#### 4.3 外连接

普通等值连接 / 自然连接，**只会保留两边匹配成功的元组**；不匹配的直接丢弃。 

外连接：**保留部分不匹配的元组，缺失的属性填 NULL**。 

外连接分为三类：**左外连接、右外连接、全外连接**。

---------------

**(1) 左外连接（Left Outer Join）**

$R \bowtie_\ast S$ : 

1. 先算出自然连接 $R\bowtie S$（两边匹配成功的元组）
2. **把 $R$ 中所有没能匹配上 $S$ 的元组，全部追加进结果**；$S$ 那边缺失属性填 $\text{null}$

对于选课的例子，如果希望：

> **显示所有学生，包括没有选课的学生。**

就需要使用**左外连接（Left Outer Join）**。它保留左侧关系的所有元组；找不到匹配时，右侧属性用 `NULL` 填充：

| sid  | name | cid  | grade |
| ---- | ---- | ---- | ----- |
| S4   | 赵六 | NULL | NULL  |

SQL：

```sql
SELECT s.sid, s.name, e.cid, e.grade
FROM Student AS s
LEFT JOIN Enrollment AS e
  on s.sid = e.sid
```

**(2) 右外连接（Right Outer Join）**

$R \ast\bowtie S$：

1. 先算出自然连接 $R\bowtie S$
2. **保留 S 中所有无法匹配 R 的元组**，R 缺失属性填 NULL

左外连接与有外连接的关系：

$$
R \ast\bowtie S = S \bowtie_\ast R
$$

**(3) 全外连接（Full Outer Join）**

$R \ast\bowtie_\ast S$：

1. 自然连接匹配成功的元组保留
2. $R$ 中不匹配的元组保留（$S$ 侧 NULL）
3. $S$ 中不匹配的元组保留（$R$ 侧 NULL） 两边所有元组都尽量保留，匹配不上的地方补 $\text{null}$。

SQL：

```sql
SELECT *
FROM Student FULL OUTER JOIN Enrollment
ON Student.sid = Enrollment.sid
```

-------------------------------

### 5. 除法

除法是用来解决**全称量词**类查询：找出 R 里，**和 S 中每一个元组都配对成功**的那些值。 

典型问题：**找出选修了所有指定课程的学生。**。

设：

$$
R(sid,cid)=\pi_{sid,cid}(Enrollment),
$$

并有必修课程关系 $Required(cid)$：

| cid  |
| ---- |
| C1   |
| C2   |

那么：

$$
R\div Required
$$

返回选修了 C1 和 C2 的学生，即：

| sid  |
| ---- |
| S1   |

---------------------

一般地，设 $R(X,Y)$ 与 $S(Y)$，且属性集合 $X,Y$ 不相交，则：
$$
\boxed{ R\div S = \{x\in\pi_X(R): \forall y\in S,\ (x,y)\in R\} }. 
$$

这里最关键的是全称量词：

$$
\forall y\in S.
$$

它表示候选 $x$ 必须与 $S$ 中的每一个 $y$ 都形成一个出现在 $R$ 中的组合。

除法可以用基本运算表达：

$$
\boxed{ R\div S = \pi_X(R) - \pi_X\left[ \left(\pi_X(R)\times S\right)-R \right] }.
$$

这个公式可以逐步理解：

1. $\pi_X(R)$：列出候选者。
2. $\pi_X(R)\times S$：生成每位候选者应该具备的全部组合。
3. 减去 $R$：找出缺失的组合。
4. 投影到 $X$：找出至少缺少一个组合的候选者。
5. 从候选集合中排除这些人。

它把“满足所有要求”转化成了“没有缺失要求”。

---------------------

对于上面的例子，$Enrollment \div Course$：**选出选了 Course 表里所有课程的学生学号 sid**。

> 逻辑句式： **选出学生 e.sid，不存在任何一门课程 c，使得该学生没有选这门课**。 

对应的 SQL：

```sql
SELECT DISTINCT sid
FROM Enrollment AS e                 -- E：逐个取出每一条选课记录，代表一名学生E.sid
WHERE NOT EXISTS (                   -- 外层 NOT EXISTS：【不存在】下面这种课程C
    SELECT * FROM Course AS c        -- C：遍历所有课程
    WHERE NOT EXISTS (               -- 内层 NOT EXISTS：【不存在】下面这条选课记录
        SELECT * FROM Enrollment e2
        WHERE e2.sid = e.sid AND e2.cid = c.cid
    )
)
```

**内层子查询**：`e.sid` 引用外层循环当前正在处理的**某一个学生**

```sql
SELECT * FROM Course AS c
WHERE NOT EXISTS (
    SELECT * FROM Enrollment e2
    WHERE e2.sid = e.sid AND e2.cid = c.cid
)
```

含义：对于**当前这个外层学生**，找出课程 C：**该学生没有选修 C**

外层 `WHERE NOT EXISTS(内层子查询)`：

```
SELECT DISTINCT sid
FROM Enrollment AS e  
WHERE NOT EXISTS ()
```

- 若有这样的课程 $C$，NOT EXISTS 假，不选出该学生 `e.sid`；
- 没有这样的课程 $C$，NOT EXISTS 真，选出该学生 `e.sid`


----------------

### 6. 扩展关系代数

经典基本关系代数没有直接提供求和、平均值、计数等数值聚合运算。实际数据库查询需要这些功能，因此通常引入扩展运算。

常用 $\gamma$ 表示分组聚合，例如：

$$
\gamma_{ cid;\ \operatorname{AVG}(grade)\rightarrow avg\_grade }(Enrollment).
$$

含义是：

1. 按 `cid` 分组。
2. 对每组计算平均成绩。
3. 将结果属性命名为 `avg_grade`。

对应 SQL：

```sql
SELECT cid, AVG(grade) AS avg_grade
FROM Enrollment
GROUP BY cid;
```

| 部分              | 含义                                                  |
| ----------------- | ----------------------------------------------------- |
| `SELECT cid`      | 查询结果中显示课程编号 `cid`                          |
| `AVG(grade)`      | 计算 `grade` 这一列的平均值                           |
| `AS avg_grade`    | 给计算出来的平均值起一个别名，叫 `avg_grade`          |
| `FROM Enrollment` | 数据来自 `Enrollment` 表，通常表示选课/成绩表         |
| `GROUP BY cid`    | 按照 `cid` 分组，也就是把同一门课程的所有记录放在一起 |

用数学语言表示，课程 $c$ 的平均成绩为：
$$
\overline g_c = \frac{ \sum_{t\in Enrollment:\,t.cid=c}t.grade }{ |\{t\in Enrollment:t.cid=c\}| }.
$$

------------

### 7. 例子：一个完整查询

现在考虑：

> 查询选修 Database 课程、成绩至少 90 分的统计学学生，返回其学号和姓名。

可以逐步构造。

第一步，筛选统计学学生：

$$
S= \sigma_{major='Statistics'}(Student).
$$

第二步，筛选 Database 课程：

$$
C= \sigma_{title='Database'}(Course). 
$$

第三步，筛选至少 90 分的选课记录：

$$
E= \sigma_{grade\ge90}(Enrollment).
$$

第四步，按学号与课程编号连接：

$$
J=S\bowtie E\bowtie C. 
$$

这里自然连接是明确的：$S,E$ 共享 `sid`，随后与 $C$ 共享 `cid`。

第五步，投影出学号和姓名：

$$
\boxed{ \pi_{sid,name} \left[ \sigma_{major='Statistics'}(Student) \bowtie \sigma_{grade\ge90}(Enrollment) \bowtie \sigma_{title='Database'}(Course) \right] }.
$$

对应 SQL：

```sql
SELECT DISTINCT s.sid, s.name
FROM Student AS s
JOIN Enrollment AS e
    ON s.sid = e.sid
JOIN Course AS c
    ON e.cid = c.cid
WHERE s.major = 'Statistics'
  AND c.title = 'Database'
  AND e.grade >= 90
```

这个例子体现了三个核心运算的分工：

- **选择**决定保留哪些记录。
- **连接**把相关记录组合起来。
- **投影**决定输出哪些属性。

---------

### 8. 为什么关系代数是查询优化的基础？

同一个查询可以有多个等价表达式。

例如，筛选统计学学生及其选课记录：

$$
\sigma_{major='Statistics'} (Student\bowtie Enrollment) 
$$

与：

$$
\left( \sigma_{major='Statistics'}(Student) \right) \bowtie Enrollment
$$

结果相同。

前者可以理解为“先连接，再筛选”；后者可以理解为“先筛选学生，再连接”。

由于条件只依赖 `Student` 的属性，因此可以把筛选提前：

$$
\boxed{ \sigma_{\theta}(R\bowtie_{\phi}S) = \sigma_{\theta}(R)\bowtie_{\phi}S } 
$$

这里要求 $\theta$ 只涉及 $R$ 的属性，且讨论的是适用该等价式的内连接情形。

如果学生表有一百万条记录，而统计学学生只有一万条，提前筛选可能显著减少连接的输入规模。这种变换称为**选择下推（Selection Pushdown）**。

另外两个常见等价式是：

$$
\sigma_{\theta_1\land\theta_2}(R) = \sigma_{\theta_1} \left(\sigma_{\theta_2}(R)\right), 
$$

以及在属性集合 $A\subseteq B$ 时：

$$
\pi_A(\pi_B(R))=\pi_A(R). 
$$

但“等价”只保证结果相同，不直接保证某一种执行方式一定更快。优化器还要结合数据量、索引与存储成本判断。

需要区分：

| 层面     | 关注内容           | 例子                         |
| -------- | ------------------ | ---------------------------- |
| 逻辑运算 | 要完成什么数据操作 | 选择、投影、连接             |
| 物理算法 | 如何具体执行该操作 | 全表扫描、索引扫描、哈希连接 |

关系代数表达的是逻辑操作。它的运算嵌套不意味着 DBMS 必须逐步生成并保存每个完整中间结果。

----------------

## 二、SQL

### 1. SQL 背景

SQL（Structured Query Language，结构化查询语言）是关系型数据库中用于**定义数据结构、查询数据和修改数据**的语言。

| 内容     | 解决的问题                 | 例子                 |
| -------- | -------------------------- | -------------------- |
| 关系模型 | 数据如何组织？             | 学生表及其属性、主键 |
| 关系代数 | 如何用逻辑运算表达查询？   | 选择、投影、连接     |
| SQL      | 如何向 DBMS 提交具体要求？ | 建表、查询、更新     |

---------

SQL 的五大组成部分

| 组成部分        | 英文全称                   | 核心功能                                               |
| :-------------- | :------------------------- | :----------------------------------------------------- |
| DDL             | Data-Definition Language   | 数据定义语言，负责定义关系模式、删除关系、修改关系模式 |
| DML             | Data Manipulation Language | 数据操纵语言，负责查询数据、插入/删除/修改元组         |
| DCL             | Data Control Language      | 数据控制语言（授权），负责指定对关系和视图的访问权限   |
| 事务控制        | Transaction Control        | 负责指定事务的开始和结束                               |
| 嵌入式/动态 SQL | Embedded/Dynamic SQL       | 定义如何将 SQL 语句嵌入通用编程语言中使用              |

> **SQL is based on bag (duplicates) not sets (no duplicates)**

------

### 2. 数据定义语言（DDL）

DDL（Data Definition Language，数据定义语言）用于定义数据库对象的结构。

- **数据库对象**：不是表里的数据，而是数据库里的“容器和规则”，比如数据库、表、视图、索引、约束等。
- **结构**：指表有哪些列、每列是什么类型、主键是谁、外键怎么关联、索引建在哪一列等。

**DDL 主要命令**

| 命令     | 作用                   | 例子                           |
| -------- | ---------------------- | ------------------------------ |
| CREATE   | 创建对象               | `CREATE TABLE`、`CREATE INDEX` |
| ALTER    | 修改对象结构           | `ALTER TABLE ADD COLUMN`       |
| DROP     | 删除对象               | `DROP TABLE`                   |
| TRUNCATE | 清空表数据但保留表结构 | `TRUNCATE TABLE`               |
| RENAME   | 重命名对象             | `RENAME TABLE`                 |

**常见数据类型**

| 类型            | 含义                                   | 适用例子       |
| --------------- | -------------------------------------- | -------------- |
| `INTEGER`       | 整数                                   | 课程学分、人数 |
| `DECIMAL(p, s)` | 精确十进制数，总位数为 p，小数位数为 s | 成绩、金额     |
| `CHAR(n)`       | 定长字符串                             | 固定长度的编码 |
| `VARCHAR(n)`    | 有最大长度限制的变长字符串             | 姓名、课程名称 |
| `DATE`          | 日期                                   | 出生日期       |

#### 2.1 创建表：`CREATE TABLE`

**基本结构**，沿用学生、课程和选课三张表：

```sql
CREATE TABLE Student(               # 表名
    sid   VARCHAR(10) PRIMARY KEY,  # 列名 数据类型 列级约束
    name  VARCHAR(50) NOT NULL,
    major VARCHAR(100)
)  
CREATE TABLE Course(
    cid    VARCHAR(10) PRIMARY KEY,
    title  VARCHAR(50) NOT NULL,
    credit INTEGER NOT NULL CHECK(credit>0)
)
CREATE TABLE Entrollment(
    sid   VARCHAR(10),
    cid   VARCHAR(10),
    grade DECIMAL(5,2),
    
    PRIMARY KEY (sid, cid),
    FOREIGN KEY (sid) REFERENCES Student(sid),
    FOREIGN KEY (cid) REFERENCES Course(cid),
    CHECK (grade BETWEEN 0 AND 100)
)
```

建表成功只意味着结构存在，此时表中还没有学生记录。

----------

#### 2.2 修改表结构：`ALTER TABLE`

`ALTER TABLE` 用于对**已存在的表**进行结构上的调整，包括：

- **添加列**（ADD）
- **删除列**（DROP COLUMN）
- **修改列的类型或约束**（MODIFY / ALTER COLUMN，语法因数据库而异）

```sql
ALTER TABLE Student
    ADD COLUMN email VARCHAR(50);
   
ALTER TABLE Course
    DROP COLUMN credit;
```

#### 2.3 删除表：`DROP TABLE`

```sql
DROP TABLE Enrollment;
```

它删除的是**整张表的定义及其数据**。与删除记录对比：

| 语句                      | 作用             | 表结构是否保留 |
| ------------------------- | ---------------- | -------------- |
| `DELETE FROM Enrollment;` | 删除表内所有记录 | 保留           |
| `DROP TABLE Enrollment;`  | 删除整张表       | 不保留         |

------------------

#### 2.4 主要完整性约束（Integrity Constraints）

- **定义**：IC 是关于数据库所有可能实例的陈述，用于防止因授权更改导致的数据一致性丢失，防止对数据库的意外损坏。
- **来源**：源于现实世界应用的需求（如：学生姓名不能为空、ID不能重复）。
- **特性**：我们可以通过检查数据库实例来发现是否**违反**了 IC，但永远无法仅通过观察实例来**推断** IC 为真（因为实例只是数据的一个子集）。

**五种主要约束类型**

在 `CREATE TABLE` 命令中常用的约束如下：

| 约束类型        | 说明             | 关键点                                             |
| :-------------- | :--------------- | :------------------------------------------------- |
| **NOT NULL**    | 不允许空值       | 属性必须有值。                                     |
| **PRIMARY KEY** | 主键             | 唯一标识元组，**隐含 NOT NULL**（值不能为空）。    |
| **UNIQUE**      | 唯一键（候选键） | 属性值必须唯一，但**允许为空**（NULL）。           |
| **FOREIGN KEY** | 外键             | 引用另一个关系的主键，维护表与表之间的联系。       |
| **CHECK**       | 检查约束         | 指定属性值必须满足的谓词条件（如 `credits > 0`）。 |

----------

##### （1）主键（Primary Key）与候选键（Candidate Key）

**候选键**：能唯一标识元组的最小属性集。可以有多个候选键。

**主键**：从所有候选键中选出一个作为主键。主键的值不能重复，也不能为 NULL。

**唯一键（Unique Key）**：除了主键之外的其他候选键，用 `UNIQUE` 声明。

```sql
CREATE TABLE Student(
    sid   VARCHAR(10) PRIMARY KEY,    -- 主键，自动 NOT NULL
    name  VARCHAR(50) NOT NULL,
    major VARCHAR(100),
    email VARCHAR(15) NOT NULL,
    
    UNIQUE (email)                    -- 候选键，值必须唯一
) 
```

**为什么要选出一个主键？**

- 保证每行有唯一、明确的身份

  主键是表中最核心的标识符。它要求**唯一且不能为 NULL**，所以每一行都能被准确定位。

- 外键需要稳定、明确的引用目标

  外键通常引用另外一个表的主键。比如：

  ```sql
  FOREIGN KEY (sid) REFERENCES Student(sid)
  ```

  如果 `Student`没有一个清晰的主键，其他表就不知道该引用哪一个属性来建立关系

- 主键通常决定数据的物理存储方式

  在 MySQL 的 InnoDB 引擎中，主键默认会成为**聚簇索引**，表中的数据会**按主键顺序组织存储**。这样通过主键查询会非常快。

------------

##### （2）外键（Foreign Key）

外键用于建立**表与表之间的引用关系**。

在表 `Enrollment` 中，`sid` 必须是 `Student` 表中已存在的学号，否则插入会被**拒绝**

```sql
FOREIGN KEY (sid) REFERENCES Student(sid)
```

这保证了**参照完整性**：不会出现"选课记录中的学生不存在"的情况

**外键的级联操作（Cascade Actions）**

当被引用的主键记录被**删除**或**更新**时，数据库可以采取以下策略：

删除时（ON DELETE）：

| 策略                | 行为                                             |
| ------------------- | ------------------------------------------------ |
| `NO ACTION`（默认） | 如果有关联的外键记录，则**禁止删除**主表记录     |
| `CASCADE`           | 删除主表记录时，**同时删除**所有引用它的外键记录 |
| `SET DEFAULT`       | 将外键字段的值设为默认值                         |
| `SET NULL`          | 将外键字段的值设为 NULL                          |

> 注意：如果外键字段本身是主键的一部分（如 `Enrollment` 中的 `sid`），则 `SET DEFAULT` 和 `SET NULL` 不适用，因为主键不能为 NULL。

更新时（ON UPDATE）：

| 策略                | 行为                                             |
| :------------------ | :----------------------------------------------- |
| `NO ACTION`（默认） | 如果有关联的外键记录，则**禁止更新**主键值       |
| `CASCADE`           | 更新主表主键时，**同步更新**所有引用它的外键记录 |

对于表 `Enrollment` 的创建，一个更完整的写法：

```sql
CREATE TABLE Entrollment(
    sid   VARCHAR(10),
    cid   VARCHAR(10),
    grade DECIMAL(5,2),
    
    PRIMARY KEY (sid, cid),
    FOREIGN KEY (sid) REFERENCES Student(sid)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (cid) REFERENCES Course(cid),
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CHECK (grade BETWEEN 0 AND 100)
)
```

----------

##### （3）CHECK

| 类型         | 写在哪               | 能引用几列     | 示例                                |
| ------------ | -------------------- | -------------- | ----------------------------------- |
| 属性级 CHECK | 列定义后面           | 只能引用当前列 | `credit INTEGER CHECK (credit > 0)` |
| 域 CHECK     | `CREATE DOMAIN` 中   | 只能引用当前域 | `CHECK (CourseType IN (...))`       |
| 元组级 CHECK | 表定义末尾           | 可以引用多列   | `CHECK (major != minor)`            |
| 命名约束     | 用 `CONSTRAINT 名字` | —              | 方便 `DROP` / `ADD`                 |

```sql
-- 1. 自定义域（如果数据库支持）
CREATE DOMAIN CourseType AS VARCHAR(20)
    CHECK (CourseType IN ('Core', 'Elective', 'General'));

-- 2. 学生表
CREATE TABLE Student (
    sid   VARCHAR(10) PRIMARY KEY,
    name  VARCHAR(50) NOT NULL,
    major VARCHAR(100)
);

-- 3. 课程表：属性级 CHECK + 域
CREATE TABLE Course (
    cid    VARCHAR(10) PRIMARY KEY,
    title  VARCHAR(50) NOT NULL,
    credit INTEGER NOT NULL CHECK(credit>0)
    type   CourseType
);

-- 4. 选课表：元组级 CHECK + 命名约束
CREATE TABLE Enrollment (
    sid   VARCHAR(10),
    cid   VARCHAR(10),
    grade DECIMAL(5,2),

    PRIMARY KEY (sid, cid),
    FOREIGN KEY (sid) REFERENCES Student(sid)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (cid) REFERENCES Course(cid)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    CONSTRAINT Valid_Grade
        CHECK (grade IS NULL OR (grade BETWEEN 0 AND 100))
);

ALTER TABLE Enrollment DROP CONSTRAINT Valid_Grade;
```

------------

### 3. SQL 基本查询

基本查询解决三个问题：**从哪张表取数据、保留哪些行、输出哪些列。**

#### 3.1 查询骨架：`SELECT–FROM–WHERE`

```sql
SELECT sid, name                # 列名
FROM Student AS s               # 表名 
WHERE s.major = 'Statistics';   # 条件
```

| 子句     | 作用         | 对应关系代数             |
| -------- | ------------ | ------------------------ |
| `FROM`   | 指定数据来源 | 笛卡尔积 $\times$        |
| `WHERE`  | 筛选行       | 选择 $\sigma$            |
| `SELECT` | 指定输出列   | 投影 $\pi$，但默认不去重 |

可以按“先确定来源，再筛选行，最后取列”理解其含义。这是**逻辑上的理解顺序**，实际执行方式由 DBMS 决定。

---------

#### 3.2 条件筛选、去重与空值

##### （1）SELECT 子句（投影）

查询指定列：

```sql
SELECT sid, name
FROM Student;
```

查询所有列：

```sql
SELECT *          # * 表示所有列
FROM Student;
```

`SELECT` 也可以包含计算表达式。例如，将百分制成绩换算成十分制：

```sql
SELECT sid, cid, grade / 10.0
FROM Enrollment;
```

**查询中的计算只改变输出结果，不修改表内原始成绩。**

--------

##### （2）去重：`DISTINCT`

查询所有专业

```sql
SELECT DISTINCT major
FROM Student;
```

如果查询多列：

```sql
SELECT DISTINCT sid, cid
FROM Enrollment;
```

去重针对的是**所选列的整个组合**，并不分别对每一列去重。

------------

##### （3）条件筛选：`WHERE`

常用条件如下：

| 条件类型   | 写法                      | 含义                    |
| ---------- | ------------------------- | ----------------------- |
| 比较       | `grade >= 90`             | 成绩至少 90 分          |
| 且         | `条件1 AND 条件2`         | 两个条件都满足          |
| 或         | `条件1 OR 条件2`          | 至少一个条件满足        |
| 非         | `NOT 条件`                | 对条件取否定            |
| 范围       | `grade BETWEEN 80 AND 90` | 成绩在 80～90，包含端点 |
| 集合成员   | `cid IN ('C1', 'C2')`     | 课程编号是 C1 或 C2     |
| 字符串匹配 | `name LIKE '张%'`         | 姓名以“张”开头          |

`LIKE` 中的两个常用通配符：

| 通配符 | 含义                             | 示例    |
| ------ | -------------------------------- | ------- |
| `%`    | 任意长度的字符序列，包括零个字符 | `'张%'` |
| `_`    | 恰好一个字符                     | `'张_'` |

因此，`'张%'` 匹配所有以“张”开头的姓名；`'张_'` 匹配以“张”开头、总共两个字符的姓名。

----------

##### （4）空值判断：`IS NULL`

假设 `grade` 为 `NULL` 表示成绩尚未录入。

查询尚未录入成绩的记录：

```sql
SELECT sid, cid
FROM Enrollment
WHERE grade IS NULL;
```

查询已经录入成绩的记录：

```sql
SELECT sid, cid, grade
FROM Enrollment
WHERE grade IS NOT NULL;
```

**不能使用 `grade = NULL` 判断空值。**

----------------

#### 3.3 排序与行数限制

##### （1）排序：`ORDER BY`

查询成绩从高到低排列的已出分记录（成绩相同时，按学号升序；学号也相同时，按课程编号升序。）：

```sql
SELECT sid, cid, grade
FROM Enrollment as e
WHERE grade IS NOT NULL
ORDER BY grade DESC, sid ASC, cid ASC;
```

##### （2）行数限制

```sql
SELECT sid, cid, grade
FROM Enrollment
WHERE grade IS NOT NULL
ORDER BY grade DESC, sid ASC, cid ASC
LIMIT 3;
```

--------

### 4. 连接与集合运算

这两类操作都用于组合数据，但组合方式不同：

| 操作         | 组合方式                                 | 示例                         |
| ------------ | ---------------------------------------- | ---------------------------- |
| 连接（JOIN） | 根据条件匹配两侧记录，将信息组合到同一行 | 将学生姓名与选课记录关联     |
| 集合运算     | 将列结构相容的两个查询结果求并、交或差   | 找出同时选修 C1 和 C2 的学生 |

#### 4.1 内连接、外连接与自然连接

##### （1）笛卡尔积（Cartesian Product）

`FROM` 子句中写多张表，默认做笛卡尔积（×），即**所有行的两两组合**。

```sql
SELECT *
FROM Student, Course;

# 等价于
SELECT *
FROM Student CROSS JOIN Course;
```

##### （2）自然连接（Natural Join）

自然连接使用两张表的**所有同名列**作为等值匹配条件，并**去除重复列**。

```sql
SELECT s.name, c.title, e.grade
FROM Student AS s
NATURAL JOIN Enrollment AS e NATURAL JOIN Course AS c;
```

##### （3）内连接（Inner Join）

内连接**只保留匹配成功的组合**；使用 `ON` 指定匹配条件：

```sql
SELECT s.sid, s.name, e.cid, e.grade
FROM Student AS s
INNER JOIN Enrollment AS e
    ON s.sid = e.sid
```

含义是：将每条选课记录与学号相同的学生记录组合。

关系代数中：

$$
R\bowtie_\theta S=\sigma_\theta(R\times S).
$$

逻辑上相当于从所有组合中筛选匹配项；实际执行时，DBMS 不必先生成完整的笛卡尔积。

`INNER JOIN` 通常可以简写为 `JOIN`。忘记关联条件或写错条件，可能产生大量不符合预期的组合。

##### （4）外连接（Outer Join）

外连接**保留没有匹配的记录**，没有匹配项时，右表对应的列用 `NULL` 填充。：

```sql
SELECT s.sid, s.name, e.cid, e.grade
FROM Student AS s
LEFT JOIN Enrollment AS e
    ON s.sid = e.sid
```

| 类型              | 保留哪些记录                         |
| ----------------- | ------------------------------------ |
| `INNER JOIN`      | 两侧匹配成功的组合                   |
| `LEFT JOIN`       | 匹配组合，以及左表没有匹配的记录     |
| `RIGHT JOIN`      | 匹配组合，以及右表没有匹配的记录     |
| `FULL OUTER JOIN` | 匹配组合，以及两侧各自没有匹配的记录 |

外连接中，`ON` 与 `WHERE` 的位置会影响结果：

```sql
SELECT s.sid, s.name, e.cid, e.grade
FROM Student AS s
LEFT JOIN Enrollment AS e
    ON s.sid = e.sid
WHERE e.grade >= 90;
```

先形成左连接结果，再筛选成绩。未匹配行的成绩为 `NULL`，无法通过筛选，因此会被删除。

---------

#### 4.2 并、交、差与重复值处理

##### （1）并、交、差

设两个查询结果为：

- $A$：选修 C1 的学生学号，$\{S1,S2\}$。
- $B$：选修 C2 的学生学号，$\{S1,S3\}$。

| 运算         | SQL 运算符  | 结果       | 含义               |
| ------------ | ----------- | ---------- | ------------------ |
| 并 $A\cup B$ | `UNION`     | S1、S2、S3 | 至少选了一门       |
| 交 $A\cap B$ | `INTERSECT` | S1         | 两门都选了         |
| 差 $A-B$     | `EXCEPT`    | S2         | 选了 C1，但没选 C2 |

以并运算为例：

```sql
SELECT sid
FROM Enrollment
WHERE cid = 'C1'

UNION

SELECT sid
FROM Enrollment
WHERE cid = 'C2';
```

将中间的 `UNION` 换成 `INTERSECT` 或 `EXCEPT`，就分别得到交、差。

进行集合运算时，两侧查询必须满足：

- 输出列数相同。
- 对应位置的列类型兼容。
- 列按位置对应，不按列名自动匹配。

---------------

##### （2）重复值处理：`UNION` 与 `UNION ALL`

`UNION`、`INTERSECT`、`EXCEPT` 默认采用去重语义，重复判断针对**整个输出行**。

`UNION ALL` 则保留两侧的全部行，包括重复行：

```sql
SELECT sid
FROM Enrollment
WHERE cid = 'C1'

UNION ALL

SELECT sid
FROM Enrollment
WHERE cid = 'C2';
```

本例的区别是：

| 运算        | 输出内容，不表示顺序 |
| ----------- | -------------------- |
| `UNION`     | S1、S2、S3           |
| `UNION ALL` | S1、S2、S1、S3       |

S1 在两侧各出现一次，所以 `UNION ALL` 保留两次。

选择依据是查询需求：

- 需要“不同学生名单”：使用 `UNION`。
- 需要保留所有记录及其重复次数：使用 `UNION ALL`。

-----------

### 5. 重命名

查询中可以给列或表指定临时名称，称为**别名（Alias）**。别名用于提高结果的可读性、简化 SQL，以及区分不同表中的同名列。

**别名只在相应查询范围内生效，不会修改数据库中的实际名称。**

| 别名类型 | 命名对象         | 主要用途                       |
| -------- | ---------------- | ------------------------------ |
| 列别名   | 输出列或计算结果 | 明确结果含义                   |
| 表别名   | 查询中的表引用   | 简化引用、区分来源、支持自连接 |

------------

#### 5.1 列重命名：AS 子句

基本语法：

```sql
SELECT sid AS student_id,        # SELECT 列名或表达式 AS 别名
       name AS student_name
From Student;                    # FROM 表名;
```

列别名尤其适合给**计算结果**命名：

```sql
SELECT sid,
       cid,
       grade / 10.0 AS grade_10
FROM Enrollment;
```

----------

#### 5.2 表别名：简化引用并消除歧义

基本语法：

```sql
SELECT s.sid, s.name
FROM Student AS s
WHERE s.major = 'Statistics';
```

同一张表参与查询两次时，需要用不同别名区分。

```sql
SELECT a.name as name_1,
       b.name as name_2
FROM Student AS a
JOIN Student AS b
    ON a.major = b.major
    AND a.sid < b.sid;
```

`a`、`b` 表示同一张学生表的两次引用。`a.sid < b.sid` 排除自己与自己的配对，并避免同一对学生正反各出现一次。

-------------

### 6. 聚合与分组

聚合用于**将多行数据汇总成统计结果**；分组用于**分别统计不同类别的数据**。

#### 6.1 聚合函数

聚合函数对一列数据做计算，最后返回**一个值**。

| 函数                  | 作用               | 说明         |
| :-------------------- | :----------------- | :----------- |
| `AVG(col)`            | 平均值             | 必须是数值   |
| `SUM(col)`            | 求和               | 必须是数值   |
| `MIN(col)`            | 最小值             | 可以是字符串 |
| `MAX(col)`            | 最大值             | 可以是字符串 |
| `COUNT(*)`            | 统计行数           | 包含 NULL    |
| `COUNT(col)`          | 统计非空值个数     | 忽略 NULL    |
| `COUNT(DISTINCT col)` | 统计不同非空值个数 | 去重         |
| `STDDEV(col)`         | 标准差             | 必须是数值   |

重要规则：

- `AVG`、`SUM`、`STDDEV` 的输入必须是**数值类型**。
- `MIN`、`MAX` 可以用于字符串，比如按字母顺序找最小/最大。
- 除 `COUNT(*)` 外，聚合函数都会**忽略 NULL**。

例如，统计整张选课表：

```sql
SELECT COUNT(*) AS enrollment_count,
       AVG(grade) AS avg_grade,
       MAX(grade) AS max_grade,
       STDDEV(grade) AS std_grade
FROM Enrollment;
```

----------

#### 6.2 分组：`GROUP BY`

`GROUP BY` 将指定列取值相同的分为一组，再对每个组分别使用聚合函数。

例如，统计每门课程的选课人数和平均成绩：

```sql
SELECT cid,
       COUNT(DISTINCT sid) AS enrollment_count,
       AVG(grade) AS avg_grade
FROM Enrollment
GROUP BY cid;
```

-----

#### 6.3 分组筛选：`HAVING`

`HAVING` 用于对分组后的结果进行过滤，类似于针对组的 `WHERE` 子句。

例如，只显示平均成绩至少 89 分的课程：

```sql
SELECT cid, AVG(grade) AS avg_grade
FROM Enrollment
GROUP BY cid
HAVING AVG(grade) >= 89;
```

`WHERE` 与 `HAVING` 的区别：

| 子句     | 筛选对象 | 发生阶段 | 示例               |
| -------- | -------- | -------- | ------------------ |
| `WHERE`  | 原始记录 | 分组前   | `grade >= 90`      |
| `HAVING` | 分组结果 | 分组后   | `AVG(grade) >= 90` |

这两种条件含义不同：

- `WHERE grade >= 90`：先排除低于 90 分的记录，再对剩余记录统计。
- `HAVING AVG(grade) >= 90`：用组内所有已填写成绩计算平均值，再筛选平均值达标的组。

-----

#### 6.4 查询的逻辑处理顺序

对于本阶段的查询，可以按以下顺序理解：

| 顺序 | 子句            | 作用               |
| ---- | --------------- | ------------------ |
| 1    | `FROM` / `JOIN` | 确定来源并连接数据 |
| 2    | `WHERE`         | 筛选记录           |
| 3    | `GROUP BY`      | 将记录分组         |
| 4    | `HAVING`        | 筛选分组           |
| 5    | `SELECT`        | 形成输出列         |
| 6    | `DISTINCT`      | 消除重复输出行     |
| 7    | `ORDER BY`      | 排序               |
| 8    | `LIMIT` 等      | 限制返回行数       |

例如：

```sql
SELECT cid,
       COUNT(*) AS graded_count,
       AVG(grade) AS avg_grade
FROM Enrollment
WHERE grade IS NOT NULL
GROUP BY cid
HAVING COUNT(*) >= 2
ORDER BY avg_grade DESC;
```

它依次表达：

1. 读取选课表。
2. 保留已录入成绩的记录。
3. 按课程分组。
4. 保留至少有两条已出分记录的课程。
5. 输出课程编号、已出分人数和平均成绩。
6. 按平均成绩降序排列。

**逻辑处理顺序解释查询的含义；DBMS 实际执行时可以采用不同的物理计划。**

----------

