
import retro

def test_record():
    # env = retro.make(game='SpaceInvaders-Atari2600')
    # env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record='.')
    env = retro.make(game='Airstriker-Genesis', record='.')
    obs = env.reset()
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
    env.close()

def test_playback():
    movie = retro.Movie('Airstriker-Genesis-Level1-000000.bk2')
    movie.step()

    env = retro.make(
        game=movie.get_game(),
        state=None,
        # bk2s can contain any button presses, so allow everything
        use_restricted_actions=retro.Actions.ALL,
        players=movie.players,
        )
    env.initial_state = movie.get_state()
    env.reset()

    while movie.step():
        keys = []
        for p in range(movie.players):
            for i in range(env.num_buttons):
                keys.append(movie.get_key(i, p))
        env.step(keys)
        env.render()

test_record()

# Render to Video
# python -m retro.scripts.playback_movie Airstriker-Genesis-Level1-000000.bk2
