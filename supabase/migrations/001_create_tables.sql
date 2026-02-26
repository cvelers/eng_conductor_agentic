-- Run this in your Supabase SQL Editor to set up the database tables.

-- Chat threads
create table if not exists public.threads (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null default 'New chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- Chat messages
create table if not exists public.messages (
  id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references public.threads(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null default '',
  response_payload jsonb,
  created_at timestamptz not null default now()
);

-- Indexes
create index if not exists idx_threads_user on public.threads(user_id);
create index if not exists idx_messages_thread on public.messages(thread_id);

-- Row Level Security
alter table public.threads enable row level security;
alter table public.messages enable row level security;

create policy "Users can only see their own threads"
  on public.threads for select using (auth.uid() = user_id);

create policy "Users can insert their own threads"
  on public.threads for insert with check (auth.uid() = user_id);

create policy "Users can update their own threads"
  on public.threads for update using (auth.uid() = user_id);

create policy "Users can delete their own threads"
  on public.threads for delete using (auth.uid() = user_id);

create policy "Users can see messages in their threads"
  on public.messages for select using (
    thread_id in (select id from public.threads where user_id = auth.uid())
  );

create policy "Users can insert messages in their threads"
  on public.messages for insert with check (
    thread_id in (select id from public.threads where user_id = auth.uid())
  );
