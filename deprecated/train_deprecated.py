def train(self):
        # replay_memory로부터 mini batch를 받아 policy를 업데이트
        if (self.steps_taken > LEARNING_STARTS):
            # Random minibatch for every step (new method)
            minibatch = np.random.choice(len(self.replay_memory) - 1, BATCH_SIZE, False)

            state_buf = torch.randn(BATCH_SIZE, self.input_size)
            next_state_buf = torch.randn(BATCH_SIZE, self.input_size)
            ard_buf = np.zeros((BATCH_SIZE, 3))
            y = torch.randn(BATCH_SIZE, 1)
            y_pred = torch.randn(BATCH_SIZE, 1)

            criterion = F.smooth_l1_loss()
            optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)

            for i in range(BATCH_SIZE):
                r_i = minibatch[i]
                r_item = self.replay_memory[r_i]
                state = r_item[0]
                act = r_item[1]
                reward = r_item[2]
                next_state = r_item[3]
                next_state_done = r_item[4]
                state = torch.Tensor(state)
                state.view(-1, self.input_size)
                next_state = torch.Tensor(next_state)
                next_state.view(-1, self.input_size)
                state_buf[i] = state
                next_state_buf[i] = next_state
                ard_buf[i] = act, reward, next_state_done
            
            Q_pred = self.Q(state_buf)
            target_Q_pred = self.Q(next_state_buf)

            for i in range(BATCH_SIZE):
                if ard_buf[i, 2] == 1:
                    y[i] = ard_buf[i, 1]
                else:
                    y[i] = ard_buf[i, 1] + DISCOUNT_RATE * torch.max(target_Q_pred[i]).detach()
            
                act = int(ard_buf[i, 0])
                y_pred[i] = Q_pred[i, act]

            loss = criterion(y_pred, y)

            # print('loss: %f' % loss)  # For debug
            # print('loss: {}\nfc1 weight: {}\nfc2 weight: {}\nfc3 weight: {}'.format(loss, self.Q.fc1.weight.data,
            #         self.Q.fc2.weight.data, self.Q.fc3.weight.data))

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(self.Q.parameters(), 1)  # Gradient clipping
            optimizer.step()
            
            # Random minibatch for every step (original method)
            # for i in range(BATCH_SIZE):
            #     r_i = minibatch[i]
            #     r_item = self.replay_memory[r_i]
            #     state = r_item[0]
            #     act = r_item[1]
            #     reward = r_item[2]
            #     next_state = r_item[3]
            #     next_state_done = r_item[4]

            #     # Fixed target
            #     if next_state_done:  # If next state terminates
            #         y = reward
            #     else:
            #         target_Q_pred = self.target_Q(next_state)
            #         y = reward + DISCOUNT_RATE * torch.max(target_Q_pred).detach()

            #     # Gradient descent
            #     state = torch.Tensor(state)
            #     state = state.view(-1, self.input_size) 
            #     Q_pred = self.Q(state)
            #     y_pred = Q_pred[0, act]
            #     loss_fn = nn.MSELoss()
            #     loss = loss_fn(y_pred, y) / BATCH_SIZE  # Mean squared error
            #     optimizer = optim.Adam(self.Q.parameters(), lr=LEARNING_RATE)
            #     optimizer.zero_grad()
            #     loss.backward()

            #     # Gradient clamping to [-1, 1]
            #     nn.utils.clip_grad_norm_(self.Q.parameters(), 1)

            #     optimizer.step()