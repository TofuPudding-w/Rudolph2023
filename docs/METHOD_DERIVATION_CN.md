# `Iter[D_i O_all]` 实现推导

设目标态为 `|ψ>`，当前制备线路为 `C_k`，其 residual 定义为

\[
|r_k\rangle=C_k^\dagger|\psi\rangle.
\]

## 解析增长 `D_i`

对 `|r_k>` 的 MPS 副本做 χ=2 TT-SVD，得到 `|r_k^(2)>`。原 residual 本身不被
截断。Ran-canonical χ≤2 MPS 的每个张量给出一个行等距映射，把规定列嵌入 U(4)
并确定性补全。末端 U(2) 按

\[
U_{N-2}'=(I_2\otimes U_2)U_{N-2}
\]

折叠，所以解析层 `L_{k+1}` 只有 N−1 个 U(4)。它满足

\[
L_{k+1}|0^N\rangle=|r_k^{(2)}\rangle,
\quad
\langle 0|L_{k+1}^\dagger|r_k\rangle
=\langle r_k^{(2)}|r_k\rangle.
\]

候选线路应为

\[
C_{k+1}^{(0)}=C_kL_{k+1}.
\]

由于矩阵右端先作用，新层必须放在物理执行序列最前端。代码中的 `prepend_input_layer`
正是这一含义。

## 全线路优化 `O_all`

把当前线路展平为物理执行门 `U_1,...,U_M`。更新第 m 个门时，缓存

\[
|p_m\rangle=U'_{m-1}\cdots U'_1|0\rangle,
\quad
|s_m\rangle=U_{m+1}^\dagger\cdots U_M^\dagger|\psi\rangle.
\]

环境矩阵 `F_m` 被构造成

\[
\operatorname{Tr}(F_m^\dagger U_m)
=\langle s_m|U_m|p_m\rangle.
\]

若 `F_m=WΣV†`，则 polar 最优门为 `U_opt=WV†`。论文式 (6) 的阻尼更新为

\[
U_m' = U_m\left(U_m^\dagger U_{opt}\right)^r,
\qquad r=0.6.
\]

一次 sweep 按 m 从小到大执行，因此前缀使用本 sweep 已更新门，后缀使用 sweep 开始时
的未来门。这是 forward Gauss–Seidel，而不是 Jacobi。分数主幂在切口附近不保证
`r=0.6` 的每个局域步骤单调，因此实现忠实记录负增益，不偷偷改成线搜索。

在深度 k 完成 T 次 sweep 后，用优化后的整个 `C_k` 重新计算 `C_k†|ψ>`。最终 K
深度累计门更新数必须为

\[
(N-1)T\sum_{k=1}^{K}k
=(N-1)T K(K+1)/2.
\]

例如 N=12、K=8、T=10 时恰为 3960 次。
