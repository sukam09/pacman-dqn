def preprocess(self, state):
        # pacman.py의 Gamestate 클래스를 참조하여 state로부터 자유롭게 state를 preprocessing 해보세요.
        res = []
        pacman_pos = state.getPacmanPosition()
        ghost_pos = state.getGhostPosition(1)  # Max: 1
        food_num = state.getNumFood()
        if len(state.getCapsules()) == 1:
            capsule_pos = state.getCapsules()[0]
        else:
            capsule_pos = (-1, -1)
        score = state.getScore()
        scared_timer = state.getScaredTimer()
        food = state.getFood()
        food_pos = []

        # Get food position from food grid
        for i in range(7):
            for j in range(7):
                if food[i][j]:
                    food_pos.append([i, j])
        
        # Feature vector
        # res = [pacman_pos[0], pacman_pos[1], ghost_pos[0], ghost_pos[1], food_num,
        #         capsule_pos[0], capsule_pos[1], scared_timer]
        res = [pacman_pos[0], pacman_pos[1], ghost_pos[0], ghost_pos[1], food_num,
                capsule_pos[0], capsule_pos[1]]

        # Concatenate food position to feature vector
        if food_num == 2:
            res.extend([food_pos[0][0], food_pos[0][1], food_pos[1][0], food_pos[1][1]])
        elif food_num == 1:
            res.extend([food_pos[0][0], food_pos[0][1], -1, -1])
        else:
            res.extend([-1, -1, -1, -1])
        
        # dist_to_food_1 = np.sqrt((food_pos[0][0] - pacman_pos[0]) ** 2 + (food_pos[0][1] - pacman_pos[1]) ** 2)
        # dist_to_food_2 = np.sqrt((food_pos[1][0] - pacman_pos[0]) ** 2 + (food_pos[1][1] - pacman_pos[1]) ** 2)
        # res.extend([dist_to_food_1, dist_to_food_2])

        # res = minmax_scale(res)
        # self.input_size = len(res)
        # print(res)  # For debug