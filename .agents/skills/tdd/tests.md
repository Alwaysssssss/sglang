# 良好测试与不良测试

## 良好测试

**集成风格：**通过真实接口测试，而不是模拟内部部件。

```typescript
// 良好：测试可观察行为
test("用户可以使用有效购物车结算", async () => {
  const cart = createCart();
  cart.add(product);
  const result = await checkout(cart, paymentMethod);
  expect(result.status).toBe("confirmed");
});
```

特征：

- 测试用户或调用方关心的行为；
- 只使用公共接口；
- 能经受内部重构；
- 描述“是什么”，而不是“如何实现”；
- 每个测试只包含一个逻辑断言。

## 不良测试

**实现细节测试：**与内部结构耦合。

```typescript
// 不良：测试实现细节
test("结算会调用 paymentService.process", async () => {
  const mockPayment = jest.mock(paymentService);
  await checkout(cart, payment);
  expect(mockPayment.process).toHaveBeenCalledWith(cart.total);
});
```

危险信号：

- 模拟内部协作者；
- 测试私有方法；
- 断言调用次数或顺序；
- 行为未改变时，测试却因重构而失败；
- 测试名称描述“如何实现”，而不是“是什么”；
- 通过外部手段而不是接口验证。

```typescript
// 不良：绕过接口进行验证
test("createUser 会保存用户", async () => {
  await createUser({ name: "Alice" });
  const row = await db.query("SELECT * FROM users WHERE name = ?", ["Alice"]);
  expect(row).toBeDefined();
});

// 良好：通过接口验证
test("createUser 创建的用户可以被获取", async () => {
  const user = await createUser({ name: "Alice" });
  const retrieved = await getUser(user.id);
  expect(retrieved.name).toBe("Alice");
});
```

**同义反复测试：**期望值重复实现逻辑，因此测试从构造上必然通过。

```typescript
// 不良：以代码使用的相同方式重新计算期望值
test("calculateTotal 会合计明细项", () => {
  const items = [{ price: 10 }, { price: 5 }];
  const expected = items.reduce((sum, i) => sum + i.price, 0);
  expect(calculateTotal(items)).toBe(expected);
});

// 良好：期望值是独立且已知正确的字面值
test("calculateTotal 会合计明细项", () => {
  expect(calculateTotal([{ price: 10 }, { price: 5 }])).toBe(15);
});
```
