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

| 术语             | 含义                           | 例子                              |
| ---------------- | ------------------------------ | --------------------------------- |
| 关系 Relation    | 元组组成的集合，通常表示成表   | `Student`                         |
| 属性 Attribute   | 一个有名称的字段，通常表示成列 | `major`                           |
| 元组 Tuple       | 关系中的一条记录，通常表示成行 | `(S1, 张三, Statistics)`          |
| 域 Domain        | 某个属性允许取值的集合         | 成绩域可以是 \([0,100]\) 内的数值 |
| 度 Degree／Arity | 属性的数量                     | `Student` 的度为 3                |
| 基数 Cardinality | 元组的数量                     | 当前 `Student` 的基数为 4         |

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

那么 $K$ 是一个**超键（Superkey）**。这里 \(t[K]\) 表示元组 \(t\) 在属性集合 \(K\) 上的取值。

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

| 运算                       | 符号       | 作用                                   |
| -------------------------- | ---------- | -------------------------------------- |
| 选择 Selection             | \(\sigma\) | 按条件筛选元组                         |
| 投影 Projection            | \(\pi\)    | 保留指定属性                           |
| 并 Union                   | \(\cup\)   | 合并两个关系的元组                     |
| 差 Difference              | \(-\)      | 保留属于第一个但不属于第二个关系的元组 |
| 笛卡尔积 Cartesian Product | \(\times\) | 将两个关系中的元组两两组合             |
| 重命名 Rename              | \(\rho\)   | 修改关系名或属性名                     |

连接、交、除法等运算可以由基本运算表达，但因为非常常用，通常单独讨论。

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
> 如果两个不同原元组，截取 \(A_1,\dots,A_k\) 之后的值完全一样，在结果集合里只保留一份。 （关系是集合，集合元素不能重复）

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

| 表达式      | 结果             | 含义                     |
| ----------- | ---------------- | ------------------------ |
| \(A\cup B\) | \(\{S1,S2,S3\}\) | 选了 C1 或 C2 的学生     |
| \(A\cap B\) | \(\{S1\}\)       | 两门课都选了的学生       |
| \(A-B\)     | \(\{S2\}\)       | 选了 C1 但没选 C2 的学生 |
| \(B-A\)     | \(\{S3\}\)       | 选了 C2 但没选 C1 的学生 |

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

表示将关系临时命名为 \(S\)。

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

