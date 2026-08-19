# 何时使用模拟

只在**系统边界**使用模拟：

- 外部 API，例如付款、邮件；
- 外部持久化系统，且应优先使用本地测试替身；
- 时间或随机性；
- 文件系统（某些情况）。

不要模拟：

- 自有类或模块；
- 内部协作者；
- 能够控制的任何内容。

## 为可模拟性而设计

在系统边界设计易于模拟的接口：

**1. 使用依赖注入**

从外部传入依赖，而不是在内部创建：

```typescript
// 易于模拟
function processPayment(order, paymentClient) {
  return paymentClient.charge(order.total);
}

// 难以模拟
function processPayment(order) {
  const client = new StripeClient(process.env.STRIPE_KEY);
  return client.charge(order.total);
}
```

**2. 优先使用 SDK 风格接口，而不是通用获取器**

为每个外部操作创建专用函数，不要使用带条件逻辑的单个通用函数：

```typescript
// 良好：每个函数都可独立模拟
const api = {
  getUser: (id) => fetch(`/users/${id}`),
  getOrders: (userId) => fetch(`/users/${userId}/orders`),
  createOrder: (data) => fetch('/orders', { method: 'POST', body: data }),
};

// 不良：模拟内部需要条件逻辑
const api = {
  fetch: (endpoint, options) => fetch(endpoint, options),
};
```

SDK 方式意味着：

- 每个模拟只返回一种特定形状；
- 测试设置中没有条件逻辑；
- 更容易看出测试经过哪些端点；
- 每个端点都具有类型安全。
